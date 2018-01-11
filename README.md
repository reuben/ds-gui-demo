# DeepSpeech GUI demo

PyQT5 GUI to demo DeepSpeech

First, download or train a [DeepSpeech model](https://github.com/mozilla/DeepSpeech). Then clone the repository and do:

```bash
pip install -r requirements.txt
python main.py path/to/deepspeech/models/output_graph.pb
```

Optionally adjust the paths in `LM_BINARY_PATH`, `LM_TRIE_PATH` and `ALPHABET_CONFIG_PATH` if you're not using a release model.
