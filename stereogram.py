import numpy as np
import cv2
import utils

class Stereogram():

    def __init__(self, CANVAS_HEIGHT=256, DMAP_ENHANCE=False):

        self.CANVAS_HEIGHT = CANVAS_HEIGHT
        self.DMAP_ENHANCE = DMAP_ENHANCE


    def normalize_height(self, bg_tile, dmap):

        h, w, _ = bg_tile.shape
        bg_tile = cv2.resize(
            bg_tile, (int(w * self.CANVAS_HEIGHT / h), self.CANVAS_HEIGHT), cv2.INTER_CUBIC).astype(np.float32)

        h, w = dmap.shape
        dmap = cv2.resize(
            dmap, (int(w * self.CANVAS_HEIGHT / h), self.CANVAS_HEIGHT), cv2.INTER_CUBIC).astype(np.float32)

        return bg_tile, dmap


    def synthesis(self, bg_tile, dmap, BETA=0.3):

        # check dimension and convert rgb to grayscale depthmap
        if len(dmap.shape) == 3:
            dmap = np.mean(dmap, axis=-1)

        if self.DMAP_ENHANCE:
            dmap = (dmap>0)*(dmap+0.5)

        tile_h, tile_w, _ = bg_tile.shape
        H = dmap.shape[0]
        W = dmap.shape[1]

        # tile the texture
        m_repeat = int(W / bg_tile.shape[1] + 1)
        stereogram = np.tile(bg_tile, [1, m_repeat, 1])[:, 0:W, :]

        bias = int(tile_w/2)

        # <----
        y = np.arange(0, H)
        for x in range(int(W/2-tile_w/2)-1, 0, -1):
            d_L = dmap[y, x+bias]
            shift = tile_w * (1 - d_L*BETA)
            stereogram[y, x, :] = utils.get_sub_pxl_values(stereogram, ys=y, xs=x + shift)

        # ---->
        for x in range(int(W/2-tile_w/2), W):
            d_L = dmap[y, x-bias]
            shift = tile_w * (1 - d_L*BETA)
            stereogram[y, x, :] = utils.get_sub_pxl_values(stereogram, ys=y, xs=x - shift)

        return stereogram


    def compute_diffmaps(self, stereogram):

        def _x_shift(stereogram, i):
            x = np.copy(stereogram)
            return np.roll(x, i, axis=1)

        h, w, _ = stereogram.shape

        max_shift_dist = int(h / 4)
        match_window_width = int(h / 40)

        buff = 1e9 * np.ones((h, w, max_shift_dist))

        for i in range(match_window_width, max_shift_dist, 1):
            diff_map = np.abs(stereogram - _x_shift(stereogram, i))
            diff_map = np.mean(diff_map, axis=-1, dtype=np.float32)
            diff_map = cv2.blur(diff_map, (match_window_width, 1))
            buff[:, :, i] = diff_map

        return buff


    def easy_recover(self, stereogram, s_near=0.1, s_far=0.2):

        h, w, _ = stereogram.shape
        buff = self.compute_diffmaps(stereogram)
        # normalize to image width
        s_near = int(s_near * w)
        s_far = int(s_far * w)

        buff[:, :, 0:s_near] = 1e9
        dmap_recovered = np.argmin(buff[:, :, 0:s_far+1], axis=-1)
        dmap_recovered = (s_far - dmap_recovered) / s_far

        dmap_recovered[:, 0:s_far] = 0
        dmap_recovered = np.concatenate([
            dmap_recovered[:, int(s_far/2):],
            dmap_recovered[:, 0:int(s_far/2)]
        ], axis=1)

        return dmap_recovered

