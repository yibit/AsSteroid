import sys
import ipdb

def test_wsj2mix():
    sys.path.append('../../egs/wsj0-mix')
    import dataset
    ipdb.set_trace()
    wav_list = "/home/ssivasankaran/experiments/data/speech_separation/wsj0-mix/2speakers/wav8k/min/tt.sample.list"
    wav_base = "/home/ssivasankaran/experiments/data/speech_separation/wsj0-mix/2speakers/wav8k/min/tt"
    dt = dataset.WSJ2mixDataset(wav_list, wav_base, sample_rate=8000, segment=5.0)
    mix, srcs = dt.__getitem__(10)




if __name__ == '__main__':
    test_wsj2mix()

