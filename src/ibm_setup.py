"""Phase 5 step 1: save your IBM Quantum credentials locally and verify access.

This does NOT put your key in the repo. Credentials come from the git-ignored `.env` file in
the project root (copy `.env.example` to `.env` and fill it in, or edit `.env` directly) -
`QiskitRuntimeService.save_account` then writes them to ~/.qiskit/qiskit-ibm.json on your own
machine (outside this project folder), and every Phase 5 script after this one reads them from
there, never from `.env` or the repo.

Usage (from the project root, venv active, after editing .env):

    python -m src.ibm_setup

`.env` values can be overridden with real environment variables or --token/--instance if you'd
rather not use a file. Re-running this is safe: `overwrite=True` replaces any previously saved
account.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()   # reads .env in the project root into the environment, if present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("IBM_QUANTUM_TOKEN"),
                    help="API key from your IBM Quantum Platform dashboard "
                         "(default: read from IBM_QUANTUM_TOKEN env var)")
    ap.add_argument("--instance", default=os.environ.get("IBM_QUANTUM_INSTANCE"),
                    help="CRN of your Quantum service instance "
                         "(default: read from IBM_QUANTUM_INSTANCE env var)")
    args = ap.parse_args()

    placeholder = {"paste-your-api-key-here", "paste-your-instance-crn-here", ""}
    if not args.token or args.token in placeholder:
        sys.exit("No real token found. Edit .env in the project root and replace "
                 "IBM_QUANTUM_TOKEN with your actual API key (see README 'Phase 5').")
    if args.instance in placeholder:
        args.instance = None

    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=args.token,
        instance=args.instance,
        overwrite=True,
        set_as_default=True,
    )
    print("Saved credentials to your local Qiskit config (not in this repo).")

    service = QiskitRuntimeService()
    backends = service.backends()
    if not backends:
        sys.exit("Connected, but no backends are visible on this instance - "
                 "check the instance CRN and that it has Quantum access.")

    print(f"\nConnected. {len(backends)} backend(s) visible:")
    for b in backends:
        status = b.status()
        print(f"  {b.name:20s} {b.num_qubits:3d} qubits  "
              f"queue={status.pending_jobs:<4d} operational={status.operational}")

    least_busy = service.least_busy(operational=True, min_num_qubits=4)
    print(f"\nLeast busy backend with >= 4 qubits: {least_busy.name}")


if __name__ == "__main__":
    main()
