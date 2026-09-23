"""Tests for per-attempt :py:class:`.RequestCachePlan` behavior.

These tests use a recording backend and adapter to verify, for each request attempt:
* the sequence of cache operations (key generation, lookup, network send, commit)
* that the cache key is computed exactly once per attempt
* that response commits are atomic (response + redirect writes in one transaction boundary)
* that serializer failures and backend errors leave no partial cache state
* that redirects produce child plans explicitly linked to their parent plan
"""

from contextlib import contextmanager
from datetime import timedelta
from logging import getLogger
from unittest.mock import patch

import pytest
from requests import Request
from requests_mock import ANY as ANY_METHOD
from requests_mock import Adapter

from requests_cache import ALL_METHODS, CachedSession
from requests_cache.backends.base import BaseCache, DictStorage
from requests_cache.policy import RequestCachePlan

logger = getLogger(__name__)

MOCK_BASE = 'mock://plan.test'
URL_TEXT = f'{MOCK_BASE}/text'
URL_ETAG = f'{MOCK_BASE}/etag'
URL_REDIRECT = f'{MOCK_BASE}/redirect'
URL_REDIRECT_TARGET = f'{MOCK_BASE}/redirect_target'
ETAG = '"644b5b0155e6404a9cc4bd9d8b1ae730"'


class RecordingDictStorage(DictStorage):
    """In-memory storage that records writes and deletes to a shared event log"""

    def __init__(self, events, label, **kwargs):
        super().__init__(**kwargs)
        self._events = events
        self._label = label

    def __setitem__(self, key, value):
        self._events.append((f'write_{self._label}', key))
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._events.append((f'delete_{self._label}', key))
        super().__delitem__(key)


class RecordingCache(BaseCache):
    """In-memory cache backend that records all cache operations to an event log"""

    supports_transactions = True

    def __init__(self, events, **kwargs):
        super().__init__('recording', **kwargs)
        self.events = events
        self.key_count = 0
        self.responses = RecordingDictStorage(events, 'response')
        self.redirects = RecordingDictStorage(events, 'redirect')

    def create_key(self, request, **kwargs):
        self.key_count += 1
        self.events.append('create_key')
        return super().create_key(request, **kwargs)

    def get_response(self, key, default=None):
        self.events.append(('get_response', key))
        return super().get_response(key, default)

    def save_response(self, response, cache_key=None, expires=None, **kwargs):
        self.events.append(('save_response', cache_key))
        return super().save_response(response, cache_key, expires, **kwargs)

    @contextmanager
    def transaction(self):
        self.events.append('transaction_begin')
        try:
            yield
        except BaseException:
            self.events.append('transaction_rollback')
            raise
        else:
            self.events.append('transaction_commit')


class RecordingAdapter(Adapter):
    """requests-mock adapter that records network sends to a shared event log"""

    def __init__(self, events, **kwargs):
        super().__init__(**kwargs)
        self._events = events

    def send(self, request, **kwargs):
        self._events.append(('send', request.url))
        return super().send(request, **kwargs)


@pytest.fixture(scope='function')
def recording_session():
    """A CachedSession with a recording backend and recording mock adapter"""
    events = []
    session = CachedSession(backend=RecordingCache(events), allowable_methods=ALL_METHODS)
    adapter = RecordingAdapter(events)
    adapter.register_uri(ANY_METHOD, URL_TEXT, text='mock response')
    adapter.register_uri(ANY_METHOD, URL_ETAG, headers={'ETag': ETAG}, text='mock response')
    adapter.register_uri(
        ANY_METHOD, URL_REDIRECT, status_code=302, headers={'Location': URL_REDIRECT_TARGET}
    )
    adapter.register_uri(ANY_METHOD, URL_REDIRECT_TARGET, text='mock redirect target')
    session.mount('mock://', adapter)
    session.mock_adapter = adapter
    session.events = events
    yield session


def assert_atomic_commits(events):
    """Assert that all storage writes are contained within transaction boundaries, and that
    transaction boundaries are balanced.
    """
    depth = 0
    for event in events:
        if event == 'transaction_begin':
            depth += 1
        elif event in ('transaction_commit', 'transaction_rollback'):
            depth -= 1
        elif isinstance(event, tuple) and event[0].startswith(('write_', 'delete_')):
            assert depth > 0, f'Write outside transaction boundary: {event}'
    assert depth == 0, 'Unbalanced transaction boundaries'


def event_names(events):
    return [e if isinstance(e, str) else e[0] for e in events]


# Basic read/write sequences
# -----------------------------------------------------


def test_cache_miss__sequence(recording_session):
    """A cache miss should compute the key once, look up once, send once, and commit once"""
    response = recording_session.get(URL_TEXT)

    assert response.from_cache is False
    assert recording_session.cache.key_count == 1
    assert event_names(recording_session.events) == [
        'create_key',
        'get_response',
        'send',
        'save_response',
        'transaction_begin',
        'write_response',
        'transaction_commit',
    ]
    assert_atomic_commits(recording_session.events)


def test_cache_hit__sequence(recording_session):
    """A cache hit should compute the key once and look up once, with no network or writes"""
    recording_session.get(URL_TEXT)
    recording_session.events.clear()
    recording_session.cache.key_count = 0

    response = recording_session.get(URL_TEXT)

    assert response.from_cache is True
    assert recording_session.cache.key_count == 1
    assert event_names(recording_session.events) == ['create_key', 'get_response']


def test_stale_response__sequence(recording_session):
    """A stale cached response should trigger a resend and recommit, reusing the same key"""
    recording_session.settings.expire_after = timedelta(seconds=-1)
    recording_session.get(URL_TEXT)
    recording_session.events.clear()
    recording_session.cache.key_count = 0

    response = recording_session.get(URL_TEXT)

    assert response.from_cache is False
    assert recording_session.cache.key_count == 1
    assert event_names(recording_session.events) == [
        'create_key',
        'get_response',
        'send',
        'save_response',
        'transaction_begin',
        'write_response',
        'transaction_commit',
    ]
    assert_atomic_commits(recording_session.events)


def test_conditional_request__sequence(recording_session):
    """A 304 revalidation should commit the updated cached response, with the cache key
    computed only once for the attempt.
    """
    recording_session.settings.expire_after = timedelta(seconds=-1)
    recording_session.get(URL_ETAG)
    recording_session.events.clear()
    recording_session.cache.key_count = 0
    recording_session.mock_adapter.register_uri(ANY_METHOD, URL_ETAG, status_code=304)

    response = recording_session.get(URL_ETAG)

    assert response.from_cache is True and response.revalidated is True
    assert recording_session.cache.key_count == 1
    # The resent request should include validation headers from the cached response
    sent_request = recording_session.mock_adapter.request_history[-1]
    assert sent_request.headers['If-None-Match'] == ETAG
    assert event_names(recording_session.events) == [
        'create_key',
        'get_response',
        'send',
        'save_response',
        'transaction_begin',
        'write_response',
        'transaction_commit',
    ]
    assert_atomic_commits(recording_session.events)


# Redirects and plan chaining
# -----------------------------------------------------


def test_redirect__sequence(recording_session):
    """A redirect should produce one plan per attempt; each attempt computes its key once,
    and the redirect alias is committed in the same transaction as the final response.
    """
    response = recording_session.get(URL_REDIRECT)

    assert response.from_cache is False
    assert response.history
    cache = recording_session.cache
    # 3 keys: original request, redirect target request, and the redirect alias
    assert cache.key_count == 3
    assert event_names(recording_session.events) == [
        # Parent attempt (redirect source)
        'create_key',
        'get_response',
        'send',
        # Child attempt (redirect target)
        'create_key',
        'get_response',
        'send',
        # Child commit
        'save_response',
        'transaction_begin',
        'write_response',
        'transaction_commit',
        # Parent commit: final response + redirect alias in one transaction
        'save_response',
        'transaction_begin',
        'write_response',
        'create_key',
        'write_redirect',
        'transaction_commit',
    ]
    assert_atomic_commits(recording_session.events)
    # Both the source and target URLs should be retrievable from the cache
    assert recording_session.get(URL_REDIRECT).from_cache is True
    assert recording_session.get(URL_REDIRECT_TARGET).from_cache is True


def test_redirect__child_plan(recording_session):
    """Attempts spawned by redirects should produce child plans linked to their parent"""
    plans = []
    original_from_request = RequestCachePlan.from_request

    def spy_from_request(cls, request, cache, settings=None, parent=None, **kwargs):
        plan = original_from_request(request, cache, settings, parent=parent, **kwargs)
        plans.append(plan)
        return plan

    with patch.object(RequestCachePlan, 'from_request', classmethod(spy_from_request)):
        recording_session.get(URL_REDIRECT)

    assert len(plans) == 2
    parent_plan, child_plan = plans
    assert parent_plan.parent is None
    assert child_plan.parent is parent_plan
    assert child_plan.root is parent_plan
    # Each plan has its own fixed cache key and clock reading
    assert parent_plan.cache_key != child_plan.cache_key
    assert parent_plan.created_at is not None
    assert child_plan.created_at is not None


def test_plan__no_parent_for_top_level(recording_session):
    """A top-level request should have no parent plan, and the plan stack should be empty
    after the request completes.
    """
    recording_session.get(URL_TEXT)
    assert recording_session._active_plan() is None


# Streaming
# -----------------------------------------------------


def test_stream__sequence(recording_session):
    """A streaming response should not be cached before its body is read; the commit phase
    consumes the stream so the cached response contains the full body, and the returned
    response remains readable as a stream.
    """
    response = recording_session.get(URL_TEXT, stream=True)

    assert recording_session.cache.key_count == 1
    assert event_names(recording_session.events) == [
        'create_key',
        'get_response',
        'send',
        'save_response',
        'transaction_begin',
        'write_response',
        'transaction_commit',
    ]
    # The cached response contains the full body (not an unconsumed stream)
    cached_response = recording_session.cache.get_response(response.cache_key)
    assert cached_response.content == b'mock response'
    # The returned response is still readable
    assert response.content == b'mock response'


# Commit atomicity: serializer failures and backend errors
# -----------------------------------------------------


class FailingSerializer:
    """Serializer that always fails on dumps"""

    def dumps(self, value):
        raise ValueError('serialization failed')

    def loads(self, value):
        return value


def test_serializer_failure__no_partial_write(tmp_path):
    """If serialization fails during commit, neither the response nor any redirect aliases
    should be left in the cache.
    """
    db_path = str(tmp_path / 'test.sqlite')
    session = CachedSession(db_path, backend='sqlite', allowable_methods=ALL_METHODS)
    adapter = RecordingAdapter([])
    adapter.register_uri(ANY_METHOD, URL_TEXT, text='mock response')
    session.mount('mock://', adapter)
    session.cache.responses.serializer = FailingSerializer()

    with pytest.raises(ValueError):
        session.get(URL_TEXT)

    assert list(session.cache.responses.keys()) == []
    assert list(session.cache.redirects.keys()) == []
    session.close()


def test_serializer_failure__filesystem_no_empty_file(tmp_path):
    """For the filesystem backend, a serialization failure should not leave a truncated
    (empty) cache file behind.
    """
    session = CachedSession(str(tmp_path / 'cache'), backend='filesystem')
    adapter = RecordingAdapter([])
    adapter.register_uri(ANY_METHOD, URL_TEXT, text='mock response')
    session.mount('mock://', adapter)
    session.cache.responses.serializer = FailingSerializer()

    with pytest.raises(ValueError):
        session.get(URL_TEXT)

    assert list(session.cache.responses.paths()) == []
    session.close()


def test_backend_error__atomic_rollback(tmp_path):
    """With a transaction-capable backend, if a redirect alias write fails, the response
    write in the same commit should be rolled back: no partial state is left behind.
    """
    db_path = str(tmp_path / 'test.sqlite')
    session = CachedSession(db_path, backend='sqlite', allowable_methods=ALL_METHODS)
    adapter = RecordingAdapter([])
    adapter.register_uri(
        ANY_METHOD, URL_REDIRECT, status_code=302, headers={'Location': URL_REDIRECT_TARGET}
    )
    adapter.register_uri(ANY_METHOD, URL_REDIRECT_TARGET, text='mock redirect target')
    session.mount('mock://', adapter)

    with patch.object(session.cache.redirects, '_write', side_effect=RuntimeError('db error')):
        with pytest.raises(RuntimeError):
            session.get(URL_REDIRECT)

    # The redirect target (committed separately by the child attempt) is cached, but the
    # failed parent commit left neither the response nor the redirect alias
    assert len(session.cache.redirects) == 0
    target_key = session.cache.create_key(Request('GET', URL_REDIRECT_TARGET).prepare())
    assert list(session.cache.responses.keys()) == [target_key]
    session.close()


def test_backend_error__best_effort():
    """With a backend that doesn't support transactions, a failed commit retains
    best-effort semantics: earlier writes are kept.
    """
    session = CachedSession(backend='memory', allowable_methods=ALL_METHODS)
    adapter = RecordingAdapter([])
    adapter.register_uri(
        ANY_METHOD, URL_REDIRECT, status_code=302, headers={'Location': URL_REDIRECT_TARGET}
    )
    adapter.register_uri(ANY_METHOD, URL_REDIRECT_TARGET, text='mock redirect target')
    session.mount('mock://', adapter)
    assert session.cache.supports_transactions is False

    class ExplodingDictStorage(DictStorage):
        def __setitem__(self, key, value):
            raise RuntimeError('db error')

    session.cache.redirects = ExplodingDictStorage()
    with pytest.raises(RuntimeError):
        session.get(URL_REDIRECT)

    # Best-effort: the response write succeeded before the redirect alias failed
    assert len(session.cache.responses) == 2
    assert len(session.cache.redirects) == 0
    session.close()


# Clock consistency
# -----------------------------------------------------


def test_plan__consistent_clock(recording_session):
    """created_at and expires for a single request should be derived from the same clock
    reading (the plan's creation time).
    """
    recording_session.settings.expire_after = 60
    response = recording_session.get(URL_TEXT)
    cached_response = recording_session.cache.get_response(response.cache_key)

    assert cached_response.created_at == response.created_at
    assert cached_response.expires == cached_response.created_at + timedelta(seconds=60)


# Custom key functions
# -----------------------------------------------------


def test_custom_key_fn__receives_original_request():
    """A custom key function should receive the original prepared request (not the plan's
    normalized copy), preserving its public behavior.
    """
    captured = {}
    plans = []
    original_from_request = RequestCachePlan.from_request

    def key_fn(request, **kwargs):
        captured['key_request'] = request
        return 'custom_key'

    def spy_from_request(cls, request, cache, settings=None, parent=None, **kwargs):
        plan = original_from_request(request, cache, settings, parent=parent, **kwargs)
        plans.append(plan)
        return plan

    events = []
    session = CachedSession(backend='memory', key_fn=key_fn)
    adapter = RecordingAdapter(events)
    adapter.register_uri(ANY_METHOD, URL_TEXT, text='mock response')
    session.mount('mock://', adapter)

    with patch.object(RequestCachePlan, 'from_request', classmethod(spy_from_request)):
        response = session.get(URL_TEXT)

    assert response.cache_key == 'custom_key'
    assert captured['key_request'] is plans[0].request
    assert captured['key_request'] is not plans[0].normalized_request
    session.close()
