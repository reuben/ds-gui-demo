# DeepSpeech GUI demo

PyQT5 GUI to demo DeepSpeech

First, download or train a [DeepSpeech model](https://github.com/mozilla/DeepSpeech).

Then, install the deepspeech package. Right now this demo depends on an unreleased change in that package, so you should install the version from master. For example, for Python 3.6 on Linux:

```bash
pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.master.cpu/artifacts/public/deepspeech-0.1.0-cp36-cp36m-manylinux1_x86_64.whl
```

Then clone this repository and do:

```bash
pip install -r requirements.txt
python main.py path/to/deepspeech/models/output_graph.pb
```

Optionally adjust the paths in `LM_BINARY_PATH`, `LM_TRIE_PATH` and `ALPHABET_CONFIG_PATH` if you're not using a release model.
