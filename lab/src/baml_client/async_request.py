###############################################################################
#
#  Welcome to Baml! To use this generated code, please run the following:
#
#  $ pip install baml-py
#
###############################################################################

# This file was generated by BAML: please do not edit it. Instead, edit the
# BAML files and re-generate this code.
#
# ruff: noqa: E501,F401
# flake8: noqa: E501,F401
# pylint: disable=unused-import,line-too-long
# fmt: off
from typing import Any, Dict, List, Optional, Union, TypedDict, Type, Literal
from typing_extensions import NotRequired

import baml_py

from . import types
from .types import Checked, Check
from .type_builder import TypeBuilder


class BamlCallOptions(TypedDict, total=False):
    tb: NotRequired[TypeBuilder]
    client_registry: NotRequired[baml_py.baml_py.ClientRegistry]


class AsyncHttpRequest:
    __runtime: baml_py.BamlRuntime
    __ctx_manager: baml_py.BamlCtxManager

    def __init__(self, runtime: baml_py.BamlRuntime, ctx_manager: baml_py.BamlCtxManager):
      self.__runtime = runtime
      self.__ctx_manager = ctx_manager

    
    async def AskQuestion(
        self,
        question: str,docs: List[str],
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "AskQuestion",
        {
          "question": question,
          "docs": docs,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        False,
      )
    
    async def ExtractArticleMetadata(
        self,
        article: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "ExtractArticleMetadata",
        {
          "article": article,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        False,
      )
    
    async def ExtractEntities(
        self,
        question: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "ExtractEntities",
        {
          "question": question,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        False,
      )
    
    async def ExtractTriples(
        self,
        question: str,documents: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "ExtractTriples",
        {
          "question": question,
          "documents": documents,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        False,
      )
    
    async def QueryExpansion(
        self,
        question: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "QueryExpansion",
        {
          "question": question,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        False,
      )
    


class AsyncHttpStreamRequest:
    __runtime: baml_py.BamlRuntime
    __ctx_manager: baml_py.BamlCtxManager

    def __init__(self, runtime: baml_py.BamlRuntime, ctx_manager: baml_py.BamlCtxManager):
      self.__runtime = runtime
      self.__ctx_manager = ctx_manager

    
    async def AskQuestion(
        self,
        question: str,docs: List[str],
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "AskQuestion",
        {
          "question": question,
          "docs": docs,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        True,
      )
    
    async def ExtractArticleMetadata(
        self,
        article: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "ExtractArticleMetadata",
        {
          "article": article,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        True,
      )
    
    async def ExtractEntities(
        self,
        question: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "ExtractEntities",
        {
          "question": question,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        True,
      )
    
    async def ExtractTriples(
        self,
        question: str,documents: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "ExtractTriples",
        {
          "question": question,
          "documents": documents,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        True,
      )
    
    async def QueryExpansion(
        self,
        question: str,
        baml_options: BamlCallOptions = {},
    ) -> baml_py.HTTPRequest:
      __tb__ = baml_options.get("tb", None)
      if __tb__ is not None:
        tb = __tb__._tb # type: ignore (we know how to use this private attribute)
      else:
        tb = None
      __cr__ = baml_options.get("client_registry", None)

      return await self.__runtime.build_request(
        "QueryExpansion",
        {
          "question": question,
        },
        self.__ctx_manager.get(),
        tb,
        __cr__,
        True,
      )
    


__all__ = ["AsyncHttpRequest", "AsyncHttpStreamRequest"]