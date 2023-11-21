# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from datetime import datetime
import json
import shutil
import logging
import os
import sys
import time
from typing import Dict


def save_args(directory, name="cmd.txt"):
    with open(str(directory) + "/" + name, "w") as f:
        f.write(" ".join(sys.argv))


def gather_metadata() -> Dict:
    date_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Gathering git metadata.
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=None if repo.head.is_detached else repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = None
    except ImportError:
        git_data = None
    # Gathering slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(
        self,
        xpid: str = None,
        tag: str = None,
        xp_args: dict = None,
        rootdir: str = "~/logs",
        symlink_to_latest: bool = False,
        timestamp: str = None,
        use_tensorboard: bool = True,
        resume: bool = False,
        lazy_init: bool = False,
    ):
        if lazy_init:
            self.before_init = True
            return
        else:
            self.before_init = False

        if not xpid:
            # Make unique id.
            xpid = "{proc}_{unixtime}".format(proc=os.getpid(), unixtime=int(time.time()))
        self.xpid = xpid
        self.tag = tag
        self.timestamp = timestamp
        self.log_step = 0

        # Metadata gathering.
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # We need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other unwanted side-effects).
        self.metadata["args"] = copy.deepcopy(xp_args)
        self.metadata["args"]["device"] = str(self.metadata["args"]["device"])
        self.metadata["xpid"] = self.xpid
        self.metadata["tag"] = self.tag

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("logs/out")

        # To stdout handler.
        self.shandle = logging.StreamHandler()
        self.shandle.setFormatter(formatter)
        self._logger.addHandler(self.shandle)
        self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # To file handler.
        self.basepath = os.path.join(rootdir, self.xpid, f"{self.tag}_{timestamp}")

        if resume:
            if not os.path.exists(self.basepath):
                print(f"no resume path {self.basepath}")
                exit(1)
        else:
            if not os.path.exists(self.basepath):
                self._logger.info("Creating log directory: %s", self.basepath)
                os.makedirs(self.basepath, exist_ok=True)
            else:
                self._logger.info("Found log directory: %s", self.basepath)
                ans = input(
                    "log_dir is not empty. All data inside log_dir will be deleted. " "Will you proceed [y/N]? "
                )
                if ans in ["y", "Y"]:
                    shutil.rmtree(logdir)
                else:
                    exit(1)

        if symlink_to_latest:
            # Add 'latest' as symlink unless it exists and is no symlink.
            symlink = os.path.join(rootdir, "latest")
            try:
                if os.path.islink(symlink):
                    os.remove(symlink)
                if not os.path.exists(symlink):
                    os.symlink(self.basepath, symlink)
                    self._logger.info("Symlinked log directory: %s", symlink)
            except OSError:
                # os.remove() or os.symlink() raced. Don't do anything.
                pass

        self.paths = dict(
            msg="{base}/out.log".format(base=self.basepath),
            meta="{base}/meta.json".format(base=self.basepath),
        )

        self._logger.info("Saving arguments to %s", self.paths["meta"])
        if os.path.exists(self.paths["meta"]):
            self._logger.warning("Path to meta file already exists. " "Not overriding meta.")
        else:
            self._save_metadata()

        self._logger.info("Saving messages to %s", self.paths["msg"])
        if os.path.exists(self.paths["msg"]):
            self._logger.warning("Path to message file already exists. " "New data will be appended.")

        fhandle = logging.FileHandler(self.paths["msg"])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._summarywriter = SummaryWriter(self.basepath)
            self.figpath = None
        else:
            self.figpath = os.path.join(self.basepath, "fig")
            os.makedirs(self.figpath, exist_ok=True)
            self._summarywriter = None

        save_args(self.basepath, "command.txt")

    def init(self, xpid = None, tag = None, xp_args = None, rootdir = "~/logs", 
             symlink_to_latest = False, timestamp = None, use_tensorboard = True, resume = False):
        self.__init__(xpid, tag, xp_args, rootdir, symlink_to_latest, timestamp, use_tensorboard, resume, lazy_init=False)

    def log(self, string):
        if self.before_init:
            print(string)
        else:
            self._logger.info("[%s] %s" % (datetime.now(), string))

    def log_dirname(self, string):
        assert not self.before_init
        self._logger.info("%s (%s)" % (string, self.paths["msg"]))

    def scalar_summary(self, tag, value, step):
        """Add a scalar variable to summary."""
        assert not self.before_init
        if self._summarywriter is not None:
            self._summarywriter.add_scalar(f"{self.xpid}/{tag}", value, step)

    def image_summary(self, tag, image, step, dataformats="HWC"):
        """Add an image to summary."""
        assert not self.before_init
        if self._summarywriter is not None:
            self._summarywriter.add_image(f"{self.xpid}/{tag}", image, step, dataformats=dataformats)
            self._summarywriter.flush()
        else:
            pass  # TODO

    def figure_summary(self, tag, fig, step):
        assert not self.before_init
        if self._summarywriter is not None:
            self._summarywriter.add_figure(f"{self.xpid}/{tag}", fig, step)
            self._summarywriter.flush()
        else:
            fig.savefig(f"{self.figpath}/{tag}_{step:08}.png")

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        assert not self.before_init
        if self._summarywriter is not None:
            self._summarywriter.add_histogram(f"{self.xpid}/{tag}", values, step, bins="auto")

    def close(self, successful: bool = True) -> None:
        assert not self.before_init
        self.metadata["date_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.metadata["successful"] = successful
        self._save_metadata()

        for f in [self._logfile, self._fieldfile]:
            f.close()

        if self._summarywriter is not None:
            self._summarywriter.close()
        self._logger.removeHandler(self.shandle)
        del self._logger, self.shandle

    def _save_metadata(self) -> None:
        # print(self.metadata)
        def json_default(value):
            if callable(value):
                return value.__name__
            raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

        with open(self.paths["meta"], "w") as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True, default=json_default)
