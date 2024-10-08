# flake8: noqa
# type: ignore

# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""

import briton_pb2 as briton__pb2
import grpc


class BritonStub(object):
    """The greeting service definition."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Infer = channel.unary_stream(
            "/briton.Briton/Infer",
            request_serializer=briton__pb2.InferenceRequest.SerializeToString,
            response_deserializer=briton__pb2.InferenceAnswerPart.FromString,
        )


class BritonServicer(object):
    """The greeting service definition."""

    def Infer(self, request, context):
        """Sends a greeting"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_BritonServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Infer": grpc.unary_stream_rpc_method_handler(
            servicer.Infer,
            request_deserializer=briton__pb2.InferenceRequest.FromString,
            response_serializer=briton__pb2.InferenceAnswerPart.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "briton.Briton", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Briton(object):
    """The greeting service definition."""

    @staticmethod
    def Infer(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/briton.Briton/Infer",
            briton__pb2.InferenceRequest.SerializeToString,
            briton__pb2.InferenceAnswerPart.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
