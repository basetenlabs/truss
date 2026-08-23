from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Operation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OPERATION_UNSPECIFIED: _ClassVar[Operation]
    OPERATION_PUSH: _ClassVar[Operation]
    OPERATION_LIST: _ClassVar[Operation]
    OPERATION_SHOW: _ClassVar[Operation]
    OPERATION_PULL: _ClassVar[Operation]

class ErrorCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CATEGORY_UNSPECIFIED: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_INVALID_ARGUMENT: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_AUTHENTICATION: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_AUTHORIZATION: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_NOT_FOUND: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_CONFLICT: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_THROTTLED: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_QUOTA: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_UNAVAILABLE: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_INTEGRITY: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_UNSUPPORTED_PROTOCOL: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_INTERNAL: _ClassVar[ErrorCategory]

class ReferenceEntryKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REFERENCE_ENTRY_KIND_UNSPECIFIED: _ClassVar[ReferenceEntryKind]
    REFERENCE_ENTRY_KIND_TAG: _ClassVar[ReferenceEntryKind]
    REFERENCE_ENTRY_KIND_IMMUTABLE_DIGEST: _ClassVar[ReferenceEntryKind]

class FileEntryKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILE_ENTRY_KIND_UNSPECIFIED: _ClassVar[FileEntryKind]
    FILE_ENTRY_KIND_FILE: _ClassVar[FileEntryKind]
    FILE_ENTRY_KIND_DIRECTORY: _ClassVar[FileEntryKind]
    FILE_ENTRY_KIND_SYMLINK: _ClassVar[FileEntryKind]
OPERATION_UNSPECIFIED: Operation
OPERATION_PUSH: Operation
OPERATION_LIST: Operation
OPERATION_SHOW: Operation
OPERATION_PULL: Operation
ERROR_CATEGORY_UNSPECIFIED: ErrorCategory
ERROR_CATEGORY_INVALID_ARGUMENT: ErrorCategory
ERROR_CATEGORY_AUTHENTICATION: ErrorCategory
ERROR_CATEGORY_AUTHORIZATION: ErrorCategory
ERROR_CATEGORY_NOT_FOUND: ErrorCategory
ERROR_CATEGORY_CONFLICT: ErrorCategory
ERROR_CATEGORY_THROTTLED: ErrorCategory
ERROR_CATEGORY_QUOTA: ErrorCategory
ERROR_CATEGORY_UNAVAILABLE: ErrorCategory
ERROR_CATEGORY_INTEGRITY: ErrorCategory
ERROR_CATEGORY_UNSUPPORTED_PROTOCOL: ErrorCategory
ERROR_CATEGORY_INTERNAL: ErrorCategory
REFERENCE_ENTRY_KIND_UNSPECIFIED: ReferenceEntryKind
REFERENCE_ENTRY_KIND_TAG: ReferenceEntryKind
REFERENCE_ENTRY_KIND_IMMUTABLE_DIGEST: ReferenceEntryKind
FILE_ENTRY_KIND_UNSPECIFIED: FileEntryKind
FILE_ENTRY_KIND_FILE: FileEntryKind
FILE_ENTRY_KIND_DIRECTORY: FileEntryKind
FILE_ENTRY_KIND_SYMLINK: FileEntryKind

class ProtocolBootstrapV1(_message.Message):
    __slots__ = ("bootstrap_version", "cannery_version", "supported_machine_protocols", "supported_encodings")
    BOOTSTRAP_VERSION_FIELD_NUMBER: _ClassVar[int]
    CANNERY_VERSION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_MACHINE_PROTOCOLS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ENCODINGS_FIELD_NUMBER: _ClassVar[int]
    bootstrap_version: int
    cannery_version: str
    supported_machine_protocols: _containers.RepeatedScalarFieldContainer[int]
    supported_encodings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, bootstrap_version: _Optional[int] = ..., cannery_version: _Optional[str] = ..., supported_machine_protocols: _Optional[_Iterable[int]] = ..., supported_encodings: _Optional[_Iterable[str]] = ...) -> None: ...

class CommandRequestV1(_message.Message):
    __slots__ = ("protocol_version", "operation_id", "push", "list", "show", "pull")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    PUSH_FIELD_NUMBER: _ClassVar[int]
    LIST_FIELD_NUMBER: _ClassVar[int]
    SHOW_FIELD_NUMBER: _ClassVar[int]
    PULL_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    operation_id: str
    push: PushRequestV1
    list: ListRequestV1
    show: ShowRequestV1
    pull: PullRequestV1
    def __init__(self, protocol_version: _Optional[int] = ..., operation_id: _Optional[str] = ..., push: _Optional[_Union[PushRequestV1, _Mapping]] = ..., list: _Optional[_Union[ListRequestV1, _Mapping]] = ..., show: _Optional[_Union[ShowRequestV1, _Mapping]] = ..., pull: _Optional[_Union[PullRequestV1, _Mapping]] = ...) -> None: ...

class PushRequestV1(_message.Message):
    __slots__ = ("local_path", "reference", "max_bytes_in_flight", "max_concurrency")
    LOCAL_PATH_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENCY_FIELD_NUMBER: _ClassVar[int]
    local_path: str
    reference: str
    max_bytes_in_flight: int
    max_concurrency: int
    def __init__(self, local_path: _Optional[str] = ..., reference: _Optional[str] = ..., max_bytes_in_flight: _Optional[int] = ..., max_concurrency: _Optional[int] = ...) -> None: ...

class ListRequestV1(_message.Message):
    __slots__ = ("namespace_or_reference", "all", "page_size", "page_token")
    NAMESPACE_OR_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    ALL_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    namespace_or_reference: str
    all: bool
    page_size: int
    page_token: str
    def __init__(self, namespace_or_reference: _Optional[str] = ..., all: bool = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class ShowRequestV1(_message.Message):
    __slots__ = ("reference", "page_size", "page_token")
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    reference: str
    page_size: int
    page_token: str
    def __init__(self, reference: _Optional[str] = ..., page_size: _Optional[int] = ..., page_token: _Optional[str] = ...) -> None: ...

class PullRequestV1(_message.Message):
    __slots__ = ("reference", "output_directory", "max_bytes_in_flight", "max_concurrency")
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    MAX_BYTES_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENCY_FIELD_NUMBER: _ClassVar[int]
    reference: str
    output_directory: str
    max_bytes_in_flight: int
    max_concurrency: int
    def __init__(self, reference: _Optional[str] = ..., output_directory: _Optional[str] = ..., max_bytes_in_flight: _Optional[int] = ..., max_concurrency: _Optional[int] = ...) -> None: ...

class MachineRecordV1(_message.Message):
    __slots__ = ("protocol_version", "sequence", "operation_id", "operation", "started", "progress", "status", "warning", "result", "error", "cancelled")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    STARTED_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    WARNING_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CANCELLED_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    sequence: int
    operation_id: str
    operation: Operation
    started: StartedV1
    progress: ProgressV1
    status: StatusV1
    warning: WarningV1
    result: ResultV1
    error: ErrorV1
    cancelled: CancelledV1
    def __init__(self, protocol_version: _Optional[int] = ..., sequence: _Optional[int] = ..., operation_id: _Optional[str] = ..., operation: _Optional[_Union[Operation, str]] = ..., started: _Optional[_Union[StartedV1, _Mapping]] = ..., progress: _Optional[_Union[ProgressV1, _Mapping]] = ..., status: _Optional[_Union[StatusV1, _Mapping]] = ..., warning: _Optional[_Union[WarningV1, _Mapping]] = ..., result: _Optional[_Union[ResultV1, _Mapping]] = ..., error: _Optional[_Union[ErrorV1, _Mapping]] = ..., cancelled: _Optional[_Union[CancelledV1, _Mapping]] = ...) -> None: ...

class StartedV1(_message.Message):
    __slots__ = ("request", "cannery_version")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    CANNERY_VERSION_FIELD_NUMBER: _ClassVar[int]
    request: CommandRequestV1
    cannery_version: str
    def __init__(self, request: _Optional[_Union[CommandRequestV1, _Mapping]] = ..., cannery_version: _Optional[str] = ...) -> None: ...

class ProgressV1(_message.Message):
    __slots__ = ("phase", "files_done", "files_total", "bytes_done", "bytes_total", "items_done", "items_total", "elapsed_seconds")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    FILES_DONE_FIELD_NUMBER: _ClassVar[int]
    FILES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    BYTES_DONE_FIELD_NUMBER: _ClassVar[int]
    BYTES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    ITEMS_DONE_FIELD_NUMBER: _ClassVar[int]
    ITEMS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECONDS_FIELD_NUMBER: _ClassVar[int]
    phase: str
    files_done: int
    files_total: int
    bytes_done: int
    bytes_total: int
    items_done: int
    items_total: int
    elapsed_seconds: float
    def __init__(self, phase: _Optional[str] = ..., files_done: _Optional[int] = ..., files_total: _Optional[int] = ..., bytes_done: _Optional[int] = ..., bytes_total: _Optional[int] = ..., items_done: _Optional[int] = ..., items_total: _Optional[int] = ..., elapsed_seconds: _Optional[float] = ...) -> None: ...

class StatusV1(_message.Message):
    __slots__ = ("reason", "message")
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    reason: str
    message: str
    def __init__(self, reason: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class WarningV1(_message.Message):
    __slots__ = ("reason", "message")
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    reason: str
    message: str
    def __init__(self, reason: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class ResultV1(_message.Message):
    __slots__ = ("push", "list", "show", "pull")
    PUSH_FIELD_NUMBER: _ClassVar[int]
    LIST_FIELD_NUMBER: _ClassVar[int]
    SHOW_FIELD_NUMBER: _ClassVar[int]
    PULL_FIELD_NUMBER: _ClassVar[int]
    push: PushResultV1
    list: ListResultV1
    show: ShowResultV1
    pull: PullResultV1
    def __init__(self, push: _Optional[_Union[PushResultV1, _Mapping]] = ..., list: _Optional[_Union[ListResultV1, _Mapping]] = ..., show: _Optional[_Union[ShowResultV1, _Mapping]] = ..., pull: _Optional[_Union[PullResultV1, _Mapping]] = ...) -> None: ...

class PushResultV1(_message.Message):
    __slots__ = ("manifest_digest", "snapshot_sequence", "canonical_reference", "logical_bytes", "uploaded_bytes", "reused_bytes", "file_count", "directory_count", "content_created", "tag_changed")
    MANIFEST_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    UPLOADED_BYTES_FIELD_NUMBER: _ClassVar[int]
    REUSED_BYTES_FIELD_NUMBER: _ClassVar[int]
    FILE_COUNT_FIELD_NUMBER: _ClassVar[int]
    DIRECTORY_COUNT_FIELD_NUMBER: _ClassVar[int]
    CONTENT_CREATED_FIELD_NUMBER: _ClassVar[int]
    TAG_CHANGED_FIELD_NUMBER: _ClassVar[int]
    manifest_digest: str
    snapshot_sequence: int
    canonical_reference: str
    logical_bytes: int
    uploaded_bytes: int
    reused_bytes: int
    file_count: int
    directory_count: int
    content_created: bool
    tag_changed: bool
    def __init__(self, manifest_digest: _Optional[str] = ..., snapshot_sequence: _Optional[int] = ..., canonical_reference: _Optional[str] = ..., logical_bytes: _Optional[int] = ..., uploaded_bytes: _Optional[int] = ..., reused_bytes: _Optional[int] = ..., file_count: _Optional[int] = ..., directory_count: _Optional[int] = ..., content_created: bool = ..., tag_changed: bool = ...) -> None: ...

class ListResultV1(_message.Message):
    __slots__ = ("namespaces", "references")
    NAMESPACES_FIELD_NUMBER: _ClassVar[int]
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    namespaces: NamespacePageV1
    references: ReferencePageV1
    def __init__(self, namespaces: _Optional[_Union[NamespacePageV1, _Mapping]] = ..., references: _Optional[_Union[ReferencePageV1, _Mapping]] = ...) -> None: ...

class NamespacePageV1(_message.Message):
    __slots__ = ("namespaces", "next_page_token")
    NAMESPACES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    namespaces: _containers.RepeatedCompositeFieldContainer[NamespaceEntryV1]
    next_page_token: str
    def __init__(self, namespaces: _Optional[_Iterable[_Union[NamespaceEntryV1, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class NamespaceEntryV1(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class ReferencePageV1(_message.Message):
    __slots__ = ("namespace", "references", "next_page_token")
    NAMESPACE_FIELD_NUMBER: _ClassVar[int]
    REFERENCES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    namespace: str
    references: _containers.RepeatedCompositeFieldContainer[ReferenceEntryV1]
    next_page_token: str
    def __init__(self, namespace: _Optional[str] = ..., references: _Optional[_Iterable[_Union[ReferenceEntryV1, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class ReferenceEntryV1(_message.Message):
    __slots__ = ("reference", "volume", "manifest_digest", "tag", "kind", "snapshot_sequence", "created_at")
    REFERENCE_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_DIGEST_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    reference: str
    volume: str
    manifest_digest: str
    tag: str
    kind: ReferenceEntryKind
    snapshot_sequence: int
    created_at: str
    def __init__(self, reference: _Optional[str] = ..., volume: _Optional[str] = ..., manifest_digest: _Optional[str] = ..., tag: _Optional[str] = ..., kind: _Optional[_Union[ReferenceEntryKind, str]] = ..., snapshot_sequence: _Optional[int] = ..., created_at: _Optional[str] = ...) -> None: ...

class ShowResultV1(_message.Message):
    __slots__ = ("requested_reference", "canonical_reference", "manifest_digest", "snapshot_sequence", "created_at", "tags", "logical_bytes", "file_count", "directory_count", "file_page")
    REQUESTED_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    FILE_COUNT_FIELD_NUMBER: _ClassVar[int]
    DIRECTORY_COUNT_FIELD_NUMBER: _ClassVar[int]
    FILE_PAGE_FIELD_NUMBER: _ClassVar[int]
    requested_reference: str
    canonical_reference: str
    manifest_digest: str
    snapshot_sequence: int
    created_at: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    logical_bytes: int
    file_count: int
    directory_count: int
    file_page: FilePageV1
    def __init__(self, requested_reference: _Optional[str] = ..., canonical_reference: _Optional[str] = ..., manifest_digest: _Optional[str] = ..., snapshot_sequence: _Optional[int] = ..., created_at: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., logical_bytes: _Optional[int] = ..., file_count: _Optional[int] = ..., directory_count: _Optional[int] = ..., file_page: _Optional[_Union[FilePageV1, _Mapping]] = ...) -> None: ...

class FilePageV1(_message.Message):
    __slots__ = ("files", "next_page_token")
    FILES_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    files: _containers.RepeatedCompositeFieldContainer[FileEntryV1]
    next_page_token: str
    def __init__(self, files: _Optional[_Iterable[_Union[FileEntryV1, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...

class FileEntryV1(_message.Message):
    __slots__ = ("path", "kind", "size_bytes", "backing_object_digest", "symlink_target")
    PATH_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    BACKING_OBJECT_DIGEST_FIELD_NUMBER: _ClassVar[int]
    SYMLINK_TARGET_FIELD_NUMBER: _ClassVar[int]
    path: str
    kind: FileEntryKind
    size_bytes: int
    backing_object_digest: str
    symlink_target: str
    def __init__(self, path: _Optional[str] = ..., kind: _Optional[_Union[FileEntryKind, str]] = ..., size_bytes: _Optional[int] = ..., backing_object_digest: _Optional[str] = ..., symlink_target: _Optional[str] = ...) -> None: ...

class PullResultV1(_message.Message):
    __slots__ = ("requested_reference", "canonical_reference", "manifest_digest", "output_directory", "logical_bytes", "downloaded_bytes", "reused_bytes", "file_count", "directory_count", "content_verified")
    REQUESTED_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    CANONICAL_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_DIGEST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DOWNLOADED_BYTES_FIELD_NUMBER: _ClassVar[int]
    REUSED_BYTES_FIELD_NUMBER: _ClassVar[int]
    FILE_COUNT_FIELD_NUMBER: _ClassVar[int]
    DIRECTORY_COUNT_FIELD_NUMBER: _ClassVar[int]
    CONTENT_VERIFIED_FIELD_NUMBER: _ClassVar[int]
    requested_reference: str
    canonical_reference: str
    manifest_digest: str
    output_directory: str
    logical_bytes: int
    downloaded_bytes: int
    reused_bytes: int
    file_count: int
    directory_count: int
    content_verified: bool
    def __init__(self, requested_reference: _Optional[str] = ..., canonical_reference: _Optional[str] = ..., manifest_digest: _Optional[str] = ..., output_directory: _Optional[str] = ..., logical_bytes: _Optional[int] = ..., downloaded_bytes: _Optional[int] = ..., reused_bytes: _Optional[int] = ..., file_count: _Optional[int] = ..., directory_count: _Optional[int] = ..., content_verified: bool = ...) -> None: ...

class ErrorV1(_message.Message):
    __slots__ = ("category", "reason", "message", "retryable", "retry_after_ms", "details")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    RETRY_AFTER_MS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    category: ErrorCategory
    reason: str
    message: str
    retryable: bool
    retry_after_ms: int
    details: ErrorDetailsV1
    def __init__(self, category: _Optional[_Union[ErrorCategory, str]] = ..., reason: _Optional[str] = ..., message: _Optional[str] = ..., retryable: bool = ..., retry_after_ms: _Optional[int] = ..., details: _Optional[_Union[ErrorDetailsV1, _Mapping]] = ...) -> None: ...

class ErrorDetailsV1(_message.Message):
    __slots__ = ("invalid_argument", "authentication", "authorization", "not_found", "conflict", "throttled", "quota", "unavailable", "integrity", "unsupported_protocol")
    INVALID_ARGUMENT_FIELD_NUMBER: _ClassVar[int]
    AUTHENTICATION_FIELD_NUMBER: _ClassVar[int]
    AUTHORIZATION_FIELD_NUMBER: _ClassVar[int]
    NOT_FOUND_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_FIELD_NUMBER: _ClassVar[int]
    THROTTLED_FIELD_NUMBER: _ClassVar[int]
    QUOTA_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    INTEGRITY_FIELD_NUMBER: _ClassVar[int]
    UNSUPPORTED_PROTOCOL_FIELD_NUMBER: _ClassVar[int]
    invalid_argument: InvalidArgumentDetailsV1
    authentication: AuthenticationDetailsV1
    authorization: AuthorizationDetailsV1
    not_found: NotFoundDetailsV1
    conflict: ConflictDetailsV1
    throttled: ThrottledDetailsV1
    quota: QuotaDetailsV1
    unavailable: UnavailableDetailsV1
    integrity: IntegrityDetailsV1
    unsupported_protocol: UnsupportedProtocolDetailsV1
    def __init__(self, invalid_argument: _Optional[_Union[InvalidArgumentDetailsV1, _Mapping]] = ..., authentication: _Optional[_Union[AuthenticationDetailsV1, _Mapping]] = ..., authorization: _Optional[_Union[AuthorizationDetailsV1, _Mapping]] = ..., not_found: _Optional[_Union[NotFoundDetailsV1, _Mapping]] = ..., conflict: _Optional[_Union[ConflictDetailsV1, _Mapping]] = ..., throttled: _Optional[_Union[ThrottledDetailsV1, _Mapping]] = ..., quota: _Optional[_Union[QuotaDetailsV1, _Mapping]] = ..., unavailable: _Optional[_Union[UnavailableDetailsV1, _Mapping]] = ..., integrity: _Optional[_Union[IntegrityDetailsV1, _Mapping]] = ..., unsupported_protocol: _Optional[_Union[UnsupportedProtocolDetailsV1, _Mapping]] = ...) -> None: ...

class InvalidArgumentDetailsV1(_message.Message):
    __slots__ = ("field", "constraint")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINT_FIELD_NUMBER: _ClassVar[int]
    field: str
    constraint: str
    def __init__(self, field: _Optional[str] = ..., constraint: _Optional[str] = ...) -> None: ...

class AuthenticationDetailsV1(_message.Message):
    __slots__ = ("credential_kind",)
    CREDENTIAL_KIND_FIELD_NUMBER: _ClassVar[int]
    credential_kind: str
    def __init__(self, credential_kind: _Optional[str] = ...) -> None: ...

class AuthorizationDetailsV1(_message.Message):
    __slots__ = ("required_scope", "resource")
    REQUIRED_SCOPE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    required_scope: str
    resource: str
    def __init__(self, required_scope: _Optional[str] = ..., resource: _Optional[str] = ...) -> None: ...

class NotFoundDetailsV1(_message.Message):
    __slots__ = ("resource_kind", "resource")
    RESOURCE_KIND_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    resource_kind: str
    resource: str
    def __init__(self, resource_kind: _Optional[str] = ..., resource: _Optional[str] = ...) -> None: ...

class ConflictDetailsV1(_message.Message):
    __slots__ = ("resource", "current_state")
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STATE_FIELD_NUMBER: _ClassVar[int]
    resource: str
    current_state: str
    def __init__(self, resource: _Optional[str] = ..., current_state: _Optional[str] = ...) -> None: ...

class ThrottledDetailsV1(_message.Message):
    __slots__ = ("limit_name",)
    LIMIT_NAME_FIELD_NUMBER: _ClassVar[int]
    limit_name: str
    def __init__(self, limit_name: _Optional[str] = ...) -> None: ...

class QuotaDetailsV1(_message.Message):
    __slots__ = ("quota_name", "limit", "used", "requested")
    QUOTA_NAME_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    USED_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_FIELD_NUMBER: _ClassVar[int]
    quota_name: str
    limit: int
    used: int
    requested: int
    def __init__(self, quota_name: _Optional[str] = ..., limit: _Optional[int] = ..., used: _Optional[int] = ..., requested: _Optional[int] = ...) -> None: ...

class UnavailableDetailsV1(_message.Message):
    __slots__ = ("service",)
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    service: str
    def __init__(self, service: _Optional[str] = ...) -> None: ...

class IntegrityDetailsV1(_message.Message):
    __slots__ = ("path", "expected_digest", "actual_digest")
    PATH_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_DIGEST_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_DIGEST_FIELD_NUMBER: _ClassVar[int]
    path: str
    expected_digest: str
    actual_digest: str
    def __init__(self, path: _Optional[str] = ..., expected_digest: _Optional[str] = ..., actual_digest: _Optional[str] = ...) -> None: ...

class UnsupportedProtocolDetailsV1(_message.Message):
    __slots__ = ("supported_machine_protocols",)
    SUPPORTED_MACHINE_PROTOCOLS_FIELD_NUMBER: _ClassVar[int]
    supported_machine_protocols: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, supported_machine_protocols: _Optional[_Iterable[int]] = ...) -> None: ...

class CancelledV1(_message.Message):
    __slots__ = ("message", "cleanup_completed")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CLEANUP_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    message: str
    cleanup_completed: bool
    def __init__(self, message: _Optional[str] = ..., cleanup_completed: bool = ...) -> None: ...
