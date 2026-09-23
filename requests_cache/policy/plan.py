"""Per-attempt cache planning.

A :py:class:`.RequestCachePlan` bundles together everything that must stay consistent for a
single request attempt:

* the outgoing (prepared) request and its normalized form
* the cache key, computed exactly once per attempt
* a snapshot of cache policy (:py:class:`.CacheActions`)
* the candidate cached response (after lookup)
* a single clock reading used for all expiration timestamps in the attempt
* an optional link to the parent plan, for attempts spawned by redirects or retries

Network results are written to the backend in a single commit phase
(:py:meth:`.RequestCachePlan.commit`), so key generation, policy evaluation, backend lookups,
serialization, and network I/O never interleave across attempts.
"""

from __future__ import annotations

from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, Optional

from attrs import define
from requests import PreparedRequest, Response

from ..cache_keys import normalize_request
from ..models import RichMixin
from . import utcnow
from .actions import CacheActions
from .settings import CacheSettings

if TYPE_CHECKING:
    from ..backends import BaseCache
    from ..models import AnyResponse, CachedResponse

logger = getLogger(__name__)


@define(repr=False)
class RequestCachePlan(RichMixin):
    """A fixed cache plan for a single request attempt. Created via
    :py:meth:`.RequestCachePlan.from_request` at the start of :py:meth:`.CachedSession.send`.

    Args:
        request: The outgoing prepared request for this attempt
        actions: Snapshot of cache policy decisions for this attempt
        created_at: Single clock reading for this attempt; all expiration timestamps for the
            attempt are derived from it
        parent: The plan for the previous attempt, if this attempt was spawned by a
            redirect or retry
        cached_response: Candidate cached response, populated by :py:meth:`.lookup`
        normalized_request: Normalized form of the request, computed once per attempt
    """

    request: PreparedRequest
    actions: CacheActions
    created_at: datetime
    parent: Optional['RequestCachePlan'] = None
    cached_response: Optional[CachedResponse] = None
    normalized_request: Optional[PreparedRequest] = None

    @classmethod
    def from_request(
        cls,
        request: PreparedRequest,
        cache: 'BaseCache',
        settings: Optional[CacheSettings] = None,
        parent: Optional['RequestCachePlan'] = None,
        **key_kwargs,
    ) -> 'RequestCachePlan':
        """Create a plan for a new request attempt. The request is normalized and the cache
        key is computed exactly once here; later stages of the attempt reuse both.

        Args:
            request: The outgoing prepared request
            cache: Cache backend, used to generate the cache key
            settings: Session-level cache settings
            parent: Parent plan, if this attempt was spawned by a redirect or retry
            key_kwargs: Additional keyword arguments for cache key generation
        """
        settings = settings or CacheSettings()
        created_at = utcnow()
        normalized_request = normalize_request(
            request, settings.ignored_parameters, settings.content_root_key
        )
        # A custom key function receives the original request to preserve its public behavior;
        # the default key function receives the pinned normalized request (idempotent)
        key_request = request if settings.key_fn is not None else normalized_request
        cache_key = cache.create_key(key_request, **key_kwargs)
        actions = CacheActions.from_request(cache_key, request, settings, start_time=created_at)
        return cls(
            request=request,
            actions=actions,
            created_at=created_at,
            parent=parent,
            normalized_request=normalized_request,
        )

    @classmethod
    def from_actions(
        cls,
        actions: CacheActions,
        request: Optional[PreparedRequest] = None,
        parent: Optional['RequestCachePlan'] = None,
    ) -> 'RequestCachePlan':
        """Create a plan wrapping an externally constructed :py:class:`.CacheActions`, for
        internal callers that operate on actions directly.
        """
        created_at = utcnow()
        actions._start_time = actions._start_time or created_at
        return cls(
            request=request or actions._request,
            actions=actions,
            created_at=created_at,
            parent=parent,
        )

    @property
    def cache_key(self) -> str:
        """The cache key for this attempt, computed once at plan creation"""
        return self.actions.cache_key

    @property
    def expires(self):
        """Expiration time for new cached responses, derived from this attempt's clock reading"""
        return self.actions.expires

    @property
    def root(self) -> 'RequestCachePlan':
        """The first plan in the attempt chain"""
        plan = self
        while plan.parent is not None:
            plan = plan.parent
        return plan

    def lookup(self, cache: 'BaseCache', **key_kwargs) -> Optional[CachedResponse]:
        """Attempt to fetch a cached response, and update allowed actions accordingly.

        This includes the secondary ``Vary`` lookup: if the primary entry doesn't match on
        ``Vary``, try a Vary-qualified key before going to the origin server.
        """
        actions = self.actions
        cached_response: Optional[CachedResponse] = None
        if not actions.skip_read:
            cached_response = cache.get_response(actions.cache_key)
        actions.update_from_cached_response(cached_response, cache.create_key, **key_kwargs)

        if actions.vary_cache_key:
            vary_key = actions.vary_cache_key
            vary_cached = cache.get_response(vary_key)
            # Use the Vary-qualified key for any future storage (hit or miss)
            actions.cache_key = vary_key
            if vary_cached is not None:
                # Reset decision flags from the failed primary Vary check
                actions.send_request = False
                actions.resend_request = False
                actions.resend_async = False
                actions.error_504 = False
                actions.vary_cache_key = None
                actions._validation_headers = {}
                # Re-evaluate freshness/expiry with the Vary-matched response
                actions.update_from_cached_response(vary_cached, cache.create_key, **key_kwargs)
                cached_response = vary_cached

        self.cached_response = cached_response
        return cached_response

    def commit(
        self,
        cache: 'BaseCache',
        response: Response,
        cached_response: Optional[CachedResponse] = None,
    ) -> AnyResponse:
        """Commit phase for a new network response: update policy from response headers,
        serialize and write to the backend (if allowed), and wrap the response with
        cache-related attributes. Also handles conditional (304) revalidation.
        """
        from ..models import CachedResponse, OriginalResponse

        actions = self.actions
        actions.update_from_response(response)

        if not actions.skip_write:
            cache.save_response(response, self.cache_key, self.expires, created_at=self.created_at)
        elif cached_response is not None and response.status_code == 304:
            revalidated = actions.update_revalidated_response(
                response, CachedResponse.from_response(cached_response)
            )
            revalidated.cache_key = self.cache_key
            if not actions.skip_write:
                cache.save_response(revalidated, self.cache_key, self.expires)
            return revalidated
        else:
            logger.debug(f'Skipping cache write for URL: {self.request.url}')

        # This is possible if the original request is a cache miss, but updating its validation
        # headers results in redirecting to a different URL that is a cache hit
        if isinstance(response, CachedResponse):
            return response
        return OriginalResponse.wrap_response(response, actions)
