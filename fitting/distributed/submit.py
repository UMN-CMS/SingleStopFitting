from pathlib import Path
import datetime
import fitting.distributed.resources as resources

try:
    import htcondor
    import classad

    HAS_CONDOR = True
except ImportError as e:
    HAS_CONDOR = False


def getNow():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%dT%H-%M-%S")


def generateConfigFiles(configs, outdir="temp"):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)


def submitConfigs(
    exec_file=None,
    output_dir=None,
    env_path=None,
    needs_files=None,
    image="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11-cuda",
):
    log_dir = Path("logs")
    log_dir = log_dir / getNow()
    log_dir.mkdir(parents=True)

    job = htcondor.Submit(
        {
            "+SingularityImage": image,
            "executable": exec_file,
            "output": str(log_dir / "$(cluster)-$(process).out"),
            "error": str(log_dir / "$(cluster)-$(process).err"),
            "log": str(log_dir / "$(cluster)-$(process).log"),
            "request_cpus": "1",
            "request_memory": "4GB",
            "request_disk": "4GB",
            "should_transfer_files": "YES",
            "preserve_relative_paths": "true",
            "environment": f"ENVPATH={env_path} OUTDIR={output_dir}",
            "transfer_output_files": str(output_dir),
            "when_to_transfer_output": "ON_EXIT",
        }
    )
    print(job)
