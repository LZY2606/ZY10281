from collections import OrderedDict
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from requests_cache.backends import DynamoDbCache, DynamoDbDict
from tests.conftest import CACHE_NAME, fail_if_no_connection
from tests.integration.base_cache_test import BaseCacheTest
from tests.integration.base_storage_test import BaseStorageTest


AWS_OPTIONS = {
    'endpoint_url': 'http://localhost:8000',
    'region_name': 'us-east-1',
    'aws_access_key_id': 'placeholder',
    'aws_secret_access_key': 'placeholder',
}


@pytest.fixture(scope='module', autouse=True)
@fail_if_no_connection(connect_timeout=5)
def ensure_connection():
    """Fail all tests in this module if DynamoDB is not running"""
    import boto3

    client = boto3.client('dynamodb', **AWS_OPTIONS)
    client.describe_limits()


class TestDynamoDbDict(BaseStorageTest):
    storage_class = DynamoDbDict
    init_kwargs = AWS_OPTIONS

    def init_cache(self, cache_name=CACHE_NAME, index=0, clear=True, **kwargs):
        """For tests that use multiple tables, make index part of the table name"""
        kwargs = {**self.init_kwargs, **kwargs}
        cache = self.storage_class(f'{cache_name}_{index}', **kwargs)
        if clear:
            cache.clear()
        return cache

    @patch('requests_cache.backends.dynamodb.boto3.resource')
    def test_connection_kwargs(self, mock_resource):
        """A spot check to make sure optional connection kwargs gets passed to connection"""
        DynamoDbDict('test_table', region_name='us-east-2', invalid_kwarg='???')
        mock_resource.assert_called_with('dynamodb', region_name='us-east-2')

    @patch('requests_cache.backends.dynamodb.boto3.resource')
    def test_no_create_table(self, mock_resource):
        DynamoDbDict('test_table', region_name='us-east-2', create_table=False)
        assert mock_resource.create_table.call_count == 0

    def test_enable_ttl_error(self):
        """An error other than 'table already exists' should be reraised"""
        from botocore.exceptions import ClientError

        cache = self.init_cache()
        error = ClientError({'Error': {'Code': 'NullPointerException'}}, 'CreateTable')
        with patch.object(cache.connection.meta.client, 'update_time_to_live', side_effect=error):
            with pytest.raises(ClientError):
                cache._enable_ttl()

    def test_create_table_error(self):
        """An error other than 'ttl already enabled' should be reraised"""
        from botocore.exceptions import ClientError

        cache = self.init_cache()
        error = ClientError({'Error': {'Code': 'NullPointerException'}}, 'CreateTable')
        with patch.object(cache.connection, 'create_table', side_effect=error):
            with pytest.raises(ClientError):
                cache._create_table()

    @pytest.mark.parametrize('ttl_enabled', [True, False])
    def test_ttl(self, ttl_enabled):
        """DynamoDB's TTL removal process can take up to 48 hours to run, so just test if the
        'ttl' attribute is set correctly if enabled, and not set if disabled.
        """
        cache = self.init_cache(ttl=ttl_enabled)
        item = OrderedDict(foo='bar')
        item.expires_unix = 60
        cache['key'] = item

        # 'ttl' is a reserved word, so to retrieve it we need to alias it
        item = cache._table.get_item(
            Key={'key': 'key'},
            ProjectionExpression='#t',
            ExpressionAttributeNames={'#t': 'ttl'},
        )
        ttl_value = item['Item'].get('ttl')

        if ttl_enabled:
            assert isinstance(ttl_value, Decimal)
        else:
            assert ttl_value is None

    def test_iter_empty_table(self):
        cache = self.init_cache()
        cache._table.scan = MagicMock(return_value=_scan_page([]))
        assert list(cache) == []

    @pytest.mark.parametrize('use_values', [False, True], ids=['__iter__', 'values'])
    def test_pagination_all_pages_returned(self, use_values):
        cache = self.init_cache()
        cache.deserialize = lambda key, value: value
        cache._table.scan = MagicMock(
            side_effect=[
                _scan_page(['a', 'b'], last_key='b'),
                _scan_page(['c', 'd'], last_key='d'),
                _scan_page(['e']),
            ]
        )

        result = list(cache.values() if use_values else cache)

        assert result == ['a', 'b', 'c', 'd', 'e']
        assert cache._table.scan.call_count == 3

    def test_pagination_exclusive_start_key_threaded(self):
        cache = self.init_cache()
        cache._table.scan = MagicMock(
            side_effect=[
                _scan_page(['a'], last_key='a'),
                _scan_page(['b']),
            ]
        )

        list(cache)

        assert 'ExclusiveStartKey' not in cache._table.scan.call_args_list[0].kwargs
        assert cache._table.scan.call_args_list[1].kwargs['ExclusiveStartKey'] == {'key': 'a'}


def _scan_page(keys: list, last_key=None) -> dict:
    page: dict = {'Items': [{'key': k, 'value': k} for k in keys]}
    if last_key is not None:
        page['LastEvaluatedKey'] = {'key': last_key}
    return page


class TestDynamoDbCache(BaseCacheTest):
    backend_class = DynamoDbCache
    init_kwargs = AWS_OPTIONS
