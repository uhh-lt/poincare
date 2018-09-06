import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 5:
        print
        globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp1, outp2, dim = sys.argv[1:5]
    print(dim)

    model = Word2Vec(LineSentence(inp), size=int(dim), iter=15, negative=25, sg=0, min_count=5, sample=1e-6, workers=32)
    model.init_sims(replace=True)
    model.wv.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
