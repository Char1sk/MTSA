import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.utils.distance import get_distance
from src.utils.decomposition import decomposition


class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        self.distance = get_distance(args.distance)
        self.msas = args.msas
        
        self.tau = args.knn_tau
        self.m = args.knn_m
        self.knn_variant = args.knn_variant
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        if self.knn_variant == 'LSH':
            self.lsh = LSH(args)
        elif self.knn_variant == 'Decomp':
            args.knn_variant = None
            self.decomp_func = args.decomp
            self.knn_trend = TsfKNN(args)
            self.knn_season = TsfKNN(args)
            self.decomposed = False
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, :]
        if self.knn_variant == 'LSH':
            X_s = sliding_window_view(self.X, self.seq_len + self.pred_len)
            self.lsh.index(X_s, self.seq_len)

    def _search(self, x, X_s, seq_len, pred_len):
        # x: ndaaray (1, seq_len [, n_features])
        # X_s: ndarray (windows, seq_len+pred_len [, n_features])
        if self.knn_variant == 'LSH':
            # lsh = LSH(self.args)
            # lsh.index(X_s, seq_len)
            neighbor_fore = self.lsh.query(x, X_s, self.k, self.distance)
            # Use tranditional KNN for empty candidates
            if neighbor_fore.shape[0] == 0:
                distances = self.distance(x, X_s[:, :seq_len])
                indices_of_smallest_k = np.argsort(distances)[:self.k]
                neighbor_fore = X_s[indices_of_smallest_k, :]
            neighbor_fore = neighbor_fore[:, seq_len:]
            # neighbor_fore = np.concatenate([X[:,seq_len:] for X in neighbor_fore])
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore
        elif self.knn_variant == 'Decomp':
            self._get_decomp_result(X_s)
            x_trend, x_season = decomposition(x, self.decomp_func, 24)
            x_trend_fore = self.knn_trend._search(x_trend, np.concatenate((self.X_wins_trend, self.Y_wins_trend), axis=1), seq_len, pred_len)
            x_season_fore = self.knn_season._search(x_season, np.concatenate((self.X_wins_season, self.Y_wins_season), axis=1), seq_len, pred_len)
            return x_trend_fore + x_season_fore
        else:
            if self.msas == 'MIMO':
                distances = self.distance(x, X_s[:, :seq_len, :])
                indices_of_smallest_k = np.argsort(distances)[:self.k]
                neighbor_fore = X_s[indices_of_smallest_k, seq_len:, :]
                x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
                return x_fore
            elif self.msas == 'recursive': # nochange
                distances = self.distance(x, X_s[:, :seq_len])
                indices_of_smallest_k = np.argsort(distances)[:self.k]
                neighbor_fore = X_s[indices_of_smallest_k, seq_len].reshape((-1, 1))
                x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
                x_new = np.concatenate((x[:, 1:], x_fore), axis=1)
                if pred_len == 1:
                    return x_fore
                else:
                    return np.concatenate((x_fore, self._search(x_new, X_s, seq_len, pred_len - 1)), axis=1)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        # X: ndarray (windows, seq_len [, n_features])
        fore = []
        bs, seq_len, channels = X.shape
        X_s = sliding_window_view(self.X, (seq_len + pred_len, channels)).reshape(-1, seq_len + pred_len, channels)
        X_s = np.concatenate((self._lag_embed(X_s[:,:self.seq_len]), X_s[:,self.seq_len:]),axis=1)
        X = self._lag_embed(X)
        for i in range(X.shape[0]):
            x = X[i:i+1, :, :]
            x_fore = self._search(x, X_s, x.shape[1], pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore
    
    def _lag_embed(self, X):
        idx = np.arange(0, self.seq_len, self.tau)
        if self.m > 0:
            idx = idx[:self.m]
        return X[:,idx]
    
    def _get_decomp_result(self, X_s):
        if not self.decomposed:
            X_wins, Y_wins = X_s[:, :self.seq_len, :], X_s[:, self.seq_len:, :]
            self.X_wins_trend, self.X_wins_season = decomposition(X_wins, self.decomp_func, 24)
            self.Y_wins_trend, self.Y_wins_season = decomposition(Y_wins, self.decomp_func, 24)
            self.decomposed = True


class LSH():
    def __init__(self, args) -> None:
        self.hash_size  = args.hash_size
        self.input_dim  = args.seq_len
        self.num_hashes = args.num_hashes
        
        # self.hash_tables = [{} for _ in range(self.num_hashes)]
        self.BIN2DEC = np.array([2**i for i in range(self.hash_size)])
        self.hash_tables = np.ndarray((self.num_hashes, 2**self.hash_size), np.object_)
        self.uniform_vals = [np.random.randn(self.hash_size, self.input_dim) for _ in range(self.num_hashes)]
    
    def hash(self, input, uni_val):
        # input: ndarray (1, seq_len)
        sim = (input-self.means) @ uni_val.T > 0
        # idx = ''.join([str(i) for i in sim.squeeze(1).tolist()])
        return sim
    
    def hash_all(self, inputs, uni_val):
        # inputs: ndarray (windows, seq_len)
        self.means = inputs.mean(axis=0)
        sims = (inputs-self.means) @ uni_val.T > 0
        return sims
    
    def index(self, inputs, seq_len):
        # inputs: ndarray (windows, seq_len+pred_len)
        
        # for input in inputs:
        #     input = np.expand_dims(input, axis=0)
        #     for (i, uni_val) in enumerate(self.uniform_vals):
        #         self.hash_tables[i].setdefault(self.hash(input[:, :seq_len], uni_val), []).append(tuple(input.squeeze(0).tolist()))
        for (i, uni_val) in enumerate(self.uniform_vals):
            sims = self.hash_all(inputs[:, :seq_len], uni_val)
            for (j, sim) in enumerate(sims):
                # idx = ''.join([str(s) for s in sim.tolist()])
                # self.hash_tables[i].setdefault(idx, []).append(tuple(inputs[j,:].tolist()))
                # self.hash_tables[i].setdefault(idx, []).append(j)
                idx = sim @ self.BIN2DEC
                lst = self.hash_tables[i][idx]
                self.hash_tables[i][idx] = (lst if lst else []) + [j]
    
    def query(self, input, inputs, k, distance):
        # input: ndarray (1, seq_len)
        seq_len = input.shape[1]
        candidate_set = set()
        # if matches any hash-value, means neighbors
        for (i, uni_val) in enumerate(self.uniform_vals):
            # neighbors = self.hash_tables[i].get(self.hash(input, uni_val), [])
            idx = self.hash(input, uni_val).squeeze(0) @ self.BIN2DEC
            neighbors = self.hash_tables[i][idx] if self.hash_tables[i][idx] else []
            candidate_set.update(tuple(neighbors))
        # choose smallest k
        # candidate_list = [(np.array([can]), distance(input,np.array([can])[:,:seq_len])) for can in candidate_set]
        # candidate_list.sort(key=lambda x: x[1])
        # candidate_list = [pair[0] for pair in candidate_list]
        candidate_idx = list(candidate_set)
        candidate_dist = distance(input, inputs[candidate_idx, :seq_len])
        candidate_idx_arg = candidate_dist.argsort()
        candidate_idx_sort = np.array(candidate_idx, dtype=int)[candidate_idx_arg]
        # if candidate_idx_sort.size == 0: ##############
        #     print('Warn: Empty candidates')
        #     candidate_idx_sort = [0]
        candidate_list = inputs[candidate_idx_sort]
        # print(f'candidates: {len(candidate_list)}')
        return candidate_list[:k, :]
    
