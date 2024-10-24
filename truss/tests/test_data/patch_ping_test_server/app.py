import inspect
from collections import Counter

from flask import Flask

# This webserver is meant for testing the patch ping flow in truss
# Some of the end points can be used for patch ping, the names indicate
# the behavior they simulate.
# The stats endpoint can be used to collect stats for verification.

app = Flask(__name__)

_stats = Counter()


def _inc_fn_call_stat():
    global _stats
    fn_name = inspect.stack()[1][3]
    _stats[f"{fn_name}_called_count"] += 1


@app.route("/hash_is_current", methods=["POST"])
def hash_is_current():
    _inc_fn_call_stat()
    return {
        "is_current": True,
    }


@app.route("/hash_is_current_but_only_every_third_call_succeeds", methods=["POST"])
def hash_is_current_but_only_every_third_call_succeeds():
    _inc_fn_call_stat()
    global _stats
    fn_name = inspect.stack()[0][3]
    if _stats[f"{fn_name}_called_count"] % 3 == 0:
        return {
            "is_current": True,
        }
    return "simulated failure", 503


@app.route("/accepted", methods=["POST"])
def accepted():
    _inc_fn_call_stat()
    return {
        "accepted": True,
    }


@app.route("/health")
def health():
    _inc_fn_call_stat()
    return {}


@app.route("/stats")
def stats():
    global _stats
    return _stats
