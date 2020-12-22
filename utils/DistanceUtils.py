import math
# 地球半径， 单位KM
EARTH_RADIUS = 6378.137

# longitude1 第一个点经度， latitude1 第一个点纬度， longitude2第二个点经度， 第二个点纬度
def get_distance(longitude1, latitude1, longitude2, latitude2):
    # 维度
    lat1 = math.radians(latitude1)  # 角度转换为弧度
    lat2 = math.radians(latitude2)
    # 经度
    lng1 = math.radians(longitude1)
    lng2 = math.radians(longitude2)
    # 维度之差
    a = lat1 - lat2
    b = lng1 - lng2
    # 计算两点距离的公式, 其实就是计算两点之间的弧度
    distance = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat1) * math.cos(lat2) * math.pow(math.sin(b / 2), 2))
    )
    # 弧长乘以地球半径，返回单位 M
    distance = distance * EARTH_RADIUS
    return distance * 1000


# 计算两点之间的距离， 根据两点之间的经纬度
if __name__ == '__main__':
    count = get_distance(116.446227, 40.076773, 116.473478, 40.025913)
    print(count)


class distance_location:

    def __init__(self) -> None:
        super().__init__()