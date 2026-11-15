"""make_nb.py out.ipynb cell.py [cell.py ...] : first cell embeds src/*.py into /kaggle/working/src, then each cell.py is a cell."""
import json, sys, glob, os
emb = {os.path.basename(f): open(f).read() for f in glob.glob("src/*.py")}
first = "import os\nos.makedirs('/kaggle/working/src', exist_ok=True)\n" + "".join(f"open('/kaggle/working/src/{k}','w').write({v!r})\n" for k, v in emb.items()) + "import json; json.dump(%r, open('/kaggle/working/sentence_ids.json','w'))\n" % json.load(open("sentence_ids.json"))
srcs = [first] + [open(f).read() for f in sys.argv[2:]]
cells = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)} for s in srcs]
json.dump({"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}, open(sys.argv[1], "w"))
