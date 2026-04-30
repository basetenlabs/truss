"""Composable Chains — Project 2 deployable demo.

Demonstrates that ``chains.ChainletBase`` and ``chains.TrussChainlet`` coexist
in one chain: the entrypoint depends on **both** kinds at once and calls each
via its native API in the same ``run_remote``.

* ``Reverser`` — typed ``ChainletBase`` dep. Caller invokes it via the
  framework's auto-generated ``StubBase`` (``await reverser.run_remote(...)``).
* ``EchoTruss`` — ``TrussChainlet`` wrapping ``./echo_truss/`` (a plain Truss
  directory, no rewrite). Caller calls it via raw ``httpx`` using Project 1's
  descriptor helpers (``target_url`` + ``with_auth_headers``).

If the framework wires both paths correctly, the chain returns:

* ``via_chainletbase_reverser``: input reversed (typed-stub path)
* ``via_truss_chainlet_echo``: input uppercased (BYO-client path)

Both inside a single ``run_remote`` call.
"""

import httpx
import pydantic

import truss_chains as chains


class CallerOutput(pydantic.BaseModel):
    input: str
    reversed_text: str  # Reverser's output (the typed-stub call)
    reversed_then_uppercased: str  # Echo's uppercase of Reverser's output (chained)
    echo_target_url: str


_TRUSS_OVERRIDE = (
    # Override the published `truss_chains` with the Project 2 branch so
    # `chains.TrussChainlet` and the descriptor-injection code path are
    # available in every chainlet pod. Drop once shipped to PyPI.
    "truss @ git+https://github.com/basetenlabs/truss.git@matte/composable-chains-2-trusschainlet"
)


# ----- Plain Truss as a chain member ----------------------------------------


class EchoTruss(chains.TrussChainlet):
    """The user's existing Truss directory becomes a chain member with one
    declarative line. The framework copies ``./echo_truss/`` byte-for-byte
    (only ``model_metadata.chains_metadata`` is added to its config.yaml)."""

    truss_dir = "./echo_truss"


# ----- Typed ChainletBase as a chain member ---------------------------------


class Reverser(chains.ChainletBase):
    """Plain typed chainlet — exists to demonstrate that it coexists with
    ``EchoTruss`` in the same chain."""

    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi"),
        docker_image=chains.DockerImage(pip_requirements=[_TRUSS_OVERRIDE]),
    )

    async def run_remote(self, text: str) -> str:
        return text[::-1]


# ----- Entrypoint: depends on BOTH kinds simultaneously ---------------------


@chains.mark_entrypoint("Composable Chains TrussChainlet Demo")
class Caller(chains.ChainletBase):
    """Demonstrates ChainletBase / TrussChainlet co-existence by calling
    each in the same ``run_remote`` via its native API."""

    remote_config = chains.RemoteConfig(
        compute=chains.Compute(cpu_count=1, memory="512Mi"),
        docker_image=chains.DockerImage(
            pip_requirements=[_TRUSS_OVERRIDE, "httpx>=0.27"]
        ),
    )

    def __init__(
        self,
        # Typed ChainletBase dep — framework injects the auto-generated stub.
        # User invokes via ``await reverser.run_remote(text)`` (the existing path).
        reverser: Reverser = chains.depends(Reverser),
        # TrussChainlet dep — framework injects a DeployedServiceDescriptor.
        # User invokes via raw httpx using the Project 1 helpers.
        echo: chains.DeployedServiceDescriptor = chains.depends(EchoTruss),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        self._reverser = reverser
        self._echo_url = echo.target_url
        self._echo_headers = echo.with_auth_headers(context.get_baseten_api_key())

    async def run_remote(self, text: str) -> CallerOutput:
        # Step 1: typed call to the ChainletBase dep — same shape as today.
        reversed_text = await self._reverser.run_remote(text)

        # Step 2: feed Reverser's output into the TrussChainlet dep via raw httpx.
        # Demonstrates data flowing through BOTH dep flavors in one chain hop.
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                self._echo_url,
                headers=self._echo_headers,
                json={"text": reversed_text},
            )
            response.raise_for_status()
            reversed_then_uppercased = response.json()["out"]

        return CallerOutput(
            input=text,
            reversed_text=reversed_text,
            reversed_then_uppercased=reversed_then_uppercased,
            echo_target_url=self._echo_url,
        )
