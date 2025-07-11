import numpy as np
import matplotlib.pyplot as plt


def single_plot_one_curve(x: np.array, y: np.array, xlabel: str, ylabel: str, path: str):
    fig, _ = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(path.split("/")[-1].split(".")[0])
    plt.plot(x, y)
    plt.savefig(path)
    plt.close(fig)


def single_plot_multi_curves(xs: list, ys: list, xlabel: str, ylabel:str, legengds: list, path:str, xticks=None):
    fig, _ = plt.subplots()
    fig.autofmt_xdate(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(path.split("/")[-1].split(".")[0])

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        if legengds is not None:
            plt.plot(x, y,  label=legengds[i])
        else:
            plt.plot(x, y)

    if xticks is not None:
        plt.xticks(xticks)

    plt.legend()
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def single_plot_multi_curves_errbar(xs: list, ys: list, y_errs: list, xlabel: str, ylabel:str, legengds: list, path:str):
    fig, _ = plt.subplots()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(path.split("/")[-1].split(".")[0])

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        y_err = y_errs[i]
        y_min = [y[j] - y_err[j] for j in range(len(y))]
        y_max = [y[j] + y_err[j] for j in range(len(y))]

        plt.plot(x, y, label=legengds[i])
        plt.fill_between(x, y_min, y_max, alpha=0.3)

    plt.legend()
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    club = [('img_[36]&txt_[linear1]', 0.06435249745845795), ('img_[33]&txt_[linear1]', 0.061826981604099274),
            ('img_[26]&txt_[linear1]', 0.05880510061979294), ('img_[20]&txt_[linear1]', 0.05525554344058037),
            ('img_[26]&txt_[linear0]', 0.00208183447830379), ('img_[36]&txt_[linear0]', 0.0019267030293121934),
            ('img_[33]&txt_[linear0]', 0.0011127751786261797), ('img_[20]&txt_[linear0]', 0.0004388117231428623)]

    # mine = [('img_[26]&txt_[linear1]', 4.114207744598389), ('img_[20]&txt_[linear0]', 3.3086037635803223),
    #         ('img_[33]&txt_[linear1]', 2.3997316360473633), ('img_[36]&txt_[linear1]', 2.33705735206604),
    #         ('img_[33]&txt_[linear0]', 2.29941987991333), ('img_[26]&txt_[linear0]', 1.672837495803833),
    #         ('img_[36]&txt_[linear0]', 1.5861313343048096), ('img_[20]&txt_[linear1]', -11.34885311126709)]

    mine = [('img_[26]&txt_[linear1]', 4.114207744598389), ('img_[20]&txt_[linear0]', 3.3086037635803223),
            ('img_[33]&txt_[linear1]', 2.3997316360473633), ('img_[36]&txt_[linear1]', 2.33705735206604),
            ('img_[33]&txt_[linear0]', 2.29941987991333), ('img_[26]&txt_[linear0]', 1.672837495803833),
            ('img_[36]&txt_[linear0]', 1.5861313343048096), ('img_[20]&txt_[linear1]', 0)]

    # mine_label = [('img_[20]&txt_[linear0]', 3.3827309608459473), ('img_[36]&txt_[linear1]', 3.2587335109710693),
    #               ('img_[26]&txt_[linear1]', 3.013873815536499), ('img_[33]&txt_[linear1]', 1.9292185306549072),
    #               ('img_[20]&txt_[linear1]', 1.8652184009552002), ('img_[26]&txt_[linear0]', 1.7840886116027832),
    #               ('img_[33]&txt_[linear0]', 1.7465283870697021), ('img_[36]&txt_[linear0]', -13.307943344116211)]

    minejsd_label = [('img_[20]&txt_[linear0]', 3.3827309608459473), ('img_[36]&txt_[linear1]', 3.2587335109710693),
                  ('img_[26]&txt_[linear1]', 3.013873815536499), ('img_[33]&txt_[linear1]', 1.9292185306549072),
                  ('img_[20]&txt_[linear1]', 1.8652184009552002), ('img_[26]&txt_[linear0]', 1.7840886116027832),
                  ('img_[33]&txt_[linear0]', 1.7465283870697021), ('img_[36]&txt_[linear0]', 0)]

    club_label = [('img_[33]&txt_[linear1]', 1.22843337059021), ('img_[36]&txt_[linear1]', 1.2233853340148926),
                  ('img_[26]&txt_[linear1]', 1.2223131656646729), ('img_[20]&txt_[linear1]', 1.1895217895507812),
                  ('img_[26]&txt_[linear0]', 0.5613884925842285), ('img_[33]&txt_[linear0]', 0.5343278646469116),
                  ('img_[36]&txt_[linear0]', 0.5263623595237732), ('img_[20]&txt_[linear0]', 0.43979328870773315)]

    mine_label = [('img_[26]&txt_[linear1]', 1.201413631439209), ('img_[36]&txt_[linear1]', 1.1489872932434082),
                  ('img_[20]&txt_[linear1]', 1.1388216018676758), ('img_[33]&txt_[linear1]', 1.1374740600585938),
                  ('img_[33]&txt_[linear0]', 0.5674548149108887), ('img_[26]&txt_[linear0]', 0.5485525131225586),
                  ('img_[36]&txt_[linear0]', 0.5437014102935791), ('img_[20]&txt_[linear0]', 0.5026249885559082)]

    logdet_label = [('img_[20]&txt_[linear1]', 1.8070628478061899), ('img_[26]&txt_[linear1]', 1.7926932662630755),
                    ('img_[33]&txt_[linear1]', 1.7693711790287274), ('img_[36]&txt_[linear1]', 1.7290854086456058),
                    ('img_[36]&txt_[linear0]', 0.4702611718676799), ('img_[33]&txt_[linear0]', 0.45606313316016056),
                    ('img_[20]&txt_[linear0]', 0.43862591140455365), ('img_[26]&txt_[linear0]', 0.43346988328373115)]

    full_club_label = [('img_[33]&txt_[linear0]', 0.9449586868286133), ('img_[36]&txt_[linear1]', 0.914600670337677),
                       ('img_[36]&txt_[linear0]', 0.808759868144989), ('img_[26]&txt_[linear1]', 0.7040700912475586),
                       ('img_[26]&txt_[linear0]', 0.6547033786773682), ('img_[20]&txt_[linear0]', 0.33053043484687805),
                       ('img_[33]&txt_[linear1]', 4.812609404325485e-07), ('img_[20]&txt_[linear1]', -1.0381918400526047e-06)]

    full_mine_label = [('img_[36]&txt_[linear1]', 2.6991031169891357), ('img_[36]&txt_[linear0]', 2.5353424549102783),
                       ('img_[33]&txt_[linear1]', 2.448958396911621), ('img_[26]&txt_[linear1]', 2.260834217071533),
                       ('img_[33]&txt_[linear0]', 1.9794666767120361), ('img_[20]&txt_[linear1]', 1.9307523965835571),
                       ('img_[26]&txt_[linear0]', 1.896075963973999), ('img_[20]&txt_[linear0]', 1.4847712516784668)]

    full_gaussian_label = [('img_[26]&txt_[linear1]', 5.879594933600998), ('img_[20]&txt_[linear1]', 5.806475324307172),
                           ('img_[33]&txt_[linear1]', 5.677858329970292), ('img_[36]&txt_[linear1]', 5.638252470725234),
                           ('img_[26]&txt_[linear0]', 5.409891705558181), ('img_[20]&txt_[linear0]', 5.345049896637789),
                           ('img_[33]&txt_[linear0]', 5.212740270068167), ('img_[36]&txt_[linear0]', 5.168047698635888)]


    keys = ['img_[20]&txt_[linear0]', 'img_[20]&txt_[linear1]', 'img_[26]&txt_[linear0]', 'img_[26]&txt_[linear1]',
            'img_[33]&txt_[linear0]', 'img_[33]&txt_[linear1]', 'img_[36]&txt_[linear0]', 'img_[36]&txt_[linear1]']

    club_values = [dict(club_label)[key] for key in keys]
    minejsd_values = [dict(minejsd_label)[key] for key in keys]
    mine_values = [dict(mine_label)[key] for key in keys]
    logdet_values = [dict(logdet_label)[key] for key in keys]
    full_club = [dict(full_club_label)[key] for key in keys]
    full_mine = [dict(full_mine_label)[key] for key in keys]
    full_gaussian = [dict(full_gaussian_label)[key] for key in keys]

    single_plot_multi_curves([keys, keys, keys, keys, keys, keys],
                             [club_values, mine_values, logdet_values,
                              full_club, full_mine, full_gaussian],
                             xlabel="pair", ylabel="mi",
                             legengds=["club_pca", "mine_pca", "logdet_pca",
                                       "club", "mine", "logdet"], path="mi_curve_label.png")

    def vote(values):
        vote = {}
        for key in keys:
            vote[key] = 0
        max_vote = len(vote)
        for v in values:
            for index, p in enumerate(v):
                vote[p[0]] += max_vote - index

        return [(k, v) for k, v in sorted(vote.items(), key=lambda x: x[1], reverse=True)]

    print(vote([full_mine_label, full_gaussian_label]))
