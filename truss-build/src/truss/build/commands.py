from abc import abstractmethod
import shlex


# TODO(bola): Support secrets
class Command:
    @abstractmethod
    def serialize(self):
        pass


class FromCommand(Command):
    def __init__(self, image, tag=None, AS=None):
        self.image = image
        self.tag = tag
        self._as = AS

    def serialize(self):
        ret = f"FROM {self.image}"
        if self.tag is not None:
            ret += f":{self.tag}"
        if self._as is not None:
            ret += f" AS {self._as}"
        return ret


class RunCommand(Command):
    def __init__(self, command, mounts=None):
        self.command = command
        self.mounts = mounts

    def serialize(self):
        cmd = f"RUN "
        if self.mounts is not None:
            for mount in self.mounts:
                cmd += f"--mount={mount} "
        cmd += self.command
        return cmd


class CopyCommand(Command):
    # TODO(bola): From should be an image object.
    def __init__(self, src, dst, FROM=None):
        self.src = src
        self.dst = dst
        self._from = FROM

    def serialize(self):
        cmd = "COPY "
        if self._from:
            cmd += f"--from={self._from} "
        return f"{cmd}{self.src} {self.dst}"


class EntrypointCommand(Command):
    def __init__(self, command):
        self.command = command

    def serialize(self):
        return f"ENTRYPOINT {self.command}"


class EnvCommand(Command):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def serialize(self):
        return f'ENV {self.name}="{shlex.quote(self.value)}"'


class ExposeCommand(Command):
    def __init__(self, ports):
        self.ports = ports

    def serialize(self):
        return f"EXPOSE {self.ports}"


class VolumeCommand(Command):
    def __init__(self, volumes):
        self.volumes = volumes

    def serialize(self):
        return f"VOLUME {self.volumes}"


class WorkdirCommand(Command):
    def __init__(self, path):
        self.path = path

    def serialize(self):
        return f"WORKDIR {shlex.quote(self.path)}"
