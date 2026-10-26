import numpy as np

from common import settings as cfg


def wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def block_corners(position, yaw, scale=1.0):
    hx, hy = np.array(cfg.OBJECT_HALF[:2]) * scale
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return corners @ rot(yaw).T + np.asarray(position)


def fits_workspace(position, yaw):
    corners = block_corners(position, yaw)
    return bool(np.all(corners > cfg.WORKSPACE_LOW) and np.all(corners < cfg.WORKSPACE_HIGH))


def world_to_local(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])
