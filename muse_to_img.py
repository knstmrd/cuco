from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
from feature_extraction import get_features
from lightgbm import Booster
import numpy as np
import colorsys

painting_data = [{'name': 'Rothko', 'colors': ['#f8b335', '#ed6a29', '#f39434', '#fdc03e', '#fa3229']},  # 04
                 {'name': 'Monet', 'colors': ['#848aa7', '#9392a8', '#6f7ca5', '#b2918a', '#9b8d9c']},  # 03
                 {'name': 'Picasso', 'colors': ['#132f3a', '#224f5b', '#93a49c', '#103755', '#436160']},  # 05
                 {'name': 'Cuco', 'colors': ['#be8373', '#9f9646', '#8ca487', '#768cb2', '#568132']},  # 01
                 {'name': 'Bacon', 'colors': ['#562f4c', '#a42238', '#4a181c', '#ba252c', '#7d212c']}]  # 02


def hex_to_hsv(hx):
    rgb = tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


def hex_to_rgb(hx):
    rgb = tuple(int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return rgb


def process_muse_data(*args):
    # for arg in args:
    #   print(arg)
        # print(arg + '\n')

    md = args[1][0]
    lgb_b = args[1][1]
    osc_client = args[1][2]
    curr_index = md['curr_index']

    if curr_index == max_t * sampling_rate:
        curr_index = 0
        md['curr_index'] = 0

    data = args[2:]
    n_muses = len(data) // 5
    # print('received from {}'.format(n_muses))
    # print(data)
    # muse_data[curr_index, :] = [double(x) for x in data.split()]

    for i in range(n_muses):
        muse_data[i][curr_index, :] = [float(x) for x in data[i * 5: (i + 1) * 5]]
    muse_merged[curr_index, :] = muse_data[0][curr_index, :]
    for i in range(1, n_muses):
        muse_merged[curr_index, i] = muse_data[i][curr_index, i]  # take channel 'i' from person 'i'
    curr_index += 1
    md['curr_index'] += 1
    if curr_index == max_t * sampling_rate:
        output_colors = []

        X = get_samples_from_arr(muse_merged, sampling_rate, 1, 1)
        X = np.nan_to_num(X)
        Y = np.argmax(lgb_b.predict(X), axis=1)
        print('Predicted: {}'.format(painting_data[int(Y[0])]['name']))
        for c in range(5):
            oc = hex_to_rgb(painting_data[int(Y[0])]['colors'][c])
            output_colors += oc

        osc_client.send_message("/colors", output_colors)
        curr_index = 0
        md['curr_index'] = 0


def get_samples_from_arr(arr, arr_len=220, step=20, n_samples=1):
    output = np.zeros((n_samples, n_features * n_channels))
    for i in range(n_samples):
        for j in range(n_channels):
            feat_list = get_features(arr[i * step: i * step + arr_len, j])
            output[i, j * n_features:(j + 1) * n_features] = feat_list
    return output


mutable_data = {'curr_index': 0}
curr_index = 0
sampling_rate = 45
max_t = 1
n_features = 17
n_channels = 5
muse_data = [np.zeros([max_t * sampling_rate, n_channels])] * 5
muse_merged = np.zeros([max_t * sampling_rate, n_channels])
osc_ip = "10.5.0.55"
osc_port = 5005

osc_send_ip = "10.5.0.123"
osc_send_port = 5555

# lgb = Booster(model_file='lightGBM_sr_45_1sec.txt')
lgb = Booster(model_file='lightGBM_sr_45_1sec_all5_v2.txt')


client = udp_client.SimpleUDPClient(osc_send_ip, osc_send_port)

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/muse", process_muse_data, mutable_data, lgb, client)

server = osc_server.ThreadingOSCUDPServer((osc_ip, osc_port), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()
