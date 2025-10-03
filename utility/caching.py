# utility/caching.py
"""
UI-layer caching utilities for Streamlit application.

This module provides caching decorators that wrap Streamlit's cache mechanisms
without leaking UI concerns into core business logic.
"""
import streamlit as st
from functools import wraps
from typing import Callable


def cache_resource(func: Callable) -> Callable:
    """
    Cache expensive resources like DB connections and model instances.
    Wrapper around Streamlit's cache_resource.
    """
    return st.cache_resource(func)


def cache_data(ttl: int | None = None):
    """
    Cache data with optional TTL.
    Wrapper around Streamlit's cache_data.
    
    Args:
        ttl: Time-to-live in seconds (None for no expiration)
    """
    def decorator(func: Callable) -> Callable:
        if ttl:
            return st.cache_data(ttl=ttl)(func)
        return st.cache_data(func)
    return decorator


def cache_qa_chain(func: Callable) -> Callable:
    """
    Cache QA chain instances per contract.
    Uses Streamlit's cache_resource with contract_id as cache key.
    """
    @st.cache_resource
    @wraps(func)
    def wrapper(contract_id: str, *args, **kwargs):
        return func(contract_id, *args, **kwargs)
    return wrapper

