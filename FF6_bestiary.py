import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict


# class Monster(object):
#     def __init__(self, name, **kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)


def load_text(path):
    """ Note there are four enemies which have duplicate entries via same name because they appear at different points in the game with different stats.  These should be treated as separate entities. They are: Garula, Launcher (second type), Abductor, Exdeath."""

    with open(path, 'r') as f:
        for line in f:
            if "Now onto the list itself" in line:
                monster_dict = make_dict(f)
    monster_dict.pop('blank')
    return monster_dict


def make_dict(f):
    info = ['SNES Name', 'Location', 'Level', 'HP', 'MP', 'Attack', 'Defense', 'Evasion', 'Magic', 'Magic Defense', 'Magic Evasion', 'Speed', 'Gil', 'EXP', 'Steal', 'Drops', 'Elemental Immunity', 'Weak Against', 'Absorbs', 'Type', 'Status Immunity', 'Vulnerable to', 'Inherent Status', 'Other Immunity', 'Special Attack', 'Rage', 'Sketch', 'Control', 'Metamorphose', 'MP Kill', 'Imp Criticals', 'Run Difficulty', 'Attack Visual']
    monster_dict, stats_dict = {}, {}
    name = 'blank'
    for line in f:
        if len(line) > 2 and line.count('*') < 10:
            line = line.strip()
            if re.findall(r'(?<=\d\.\s)\w+', line):
                name = re.findall(r'(?<=\d.\s).*', line)[0]
                if name in monster_dict.keys():
                    previous_appearances = [re.findall(r'.*'+name+'.*', n) for n in monster_dict.keys()]
                    count = sum([len(entry) for entry in previous_appearances])
                    name += ' #{c}'.format(c=count+1)
                stats_dict['ID'] = re.findall(r'\d+', line)[0]
            elif ':' in line:
                category, values = line.split(':')[0].strip(), line.split(':')[1].strip()
                if category in info:
                    stats_dict[category] = values
            else:
                ## For multi-line entries...
                if category in info and category != 'Attack Visual':
                    stats_dict[category] += " {}".format(line)
        elif line.count('*') > 10:
            monster_dict[name] = stats_dict.copy()
            stats_dict.clear()

    if len(monster_dict[name]) == 0:
        monster_dict[name] = stats_dict.copy()

    monster_dict = clean_dict(monster_dict)
    return monster_dict


def clean_dict(monster_dict):
    """There are 8 No Name monsters which appear in code only but not in the game, as well as a few other enemies that aren't actually 'fightable', so I don't want them in this database"""
    no_names = [name for name in monster_dict.keys() if 'No Name' in name]
    not_actually_fightable = ['Kefka (Untargetable)', 'Giant', 'Kaiser Dragon #2', 'Yeti #2']
    for name in no_names + not_actually_fightable:
        monster_dict.pop(name)
    ## Kaiser Dragon has five lives of 65,500 hp each; consider setting HP to 327,500.
    return monster_dict


def make_df(monster_dict):
    df = pd.DataFrame(monster_dict).T
    df.columns = [col.replace(' ', '_') for col in df.columns.tolist()]
    df.rename(columns={'Magic': 'Magic_Attack', 'EXP': 'Exp'}, inplace=True)
    df = tidy_df(df)
    df = add_boss_col(df)
    df = add_ios_col(df)
    df = add_dragon_den_col(df)
    # df.drop(['Exp', 'Gil'], axis=1, inplace=True)
    return df


def tidy_df(df):
    for col in ['Level', 'HP', 'MP', 'Attack', 'Defense', 'Evasion', 'Magic_Attack', 'Magic_Defense', 'Magic_Evasion', 'Speed', 'Gil', 'Exp']:
        df[col] = df[col].astype(int)
    df['Rage'] = ['None' if entry is np.nan else entry for entry in df['Rage']]
    df['Vulnerable_to'] = ['None' if entry is np.nan else entry for entry in df['Vulnerable_to']]
    df.index.name = 'Name'
    return df


def add_boss_col(df):
    """ """
    bosses = ['Abyss Worm', 'Air Force', 'Angler Whelk (Head)', 'Angler Whelk (Shell)', 'Behemoth King', \
        'Behemoth King (Undead)', 'Bit', 'Blue Dragon', 'Chadarnook', 'Chadarnook (Esper)', 'Crane', \
        'Crane #2', 'Curlax', 'Dadaluma', 'Dark Behemoth', 'Deathgaze', 'Demon', 'Dullahan', 'Earth Dragon', \
        'Earth Eater', 'Erebus', 'Erebus #2', 'Erebus #3', 'Erebus #4', 'Fiend', 'Flame Eater', 'Flan Princess', \
        'Frozen Esper', 'Gargantua', 'Gigantuar', 'Gilgamesh', 'Goddess', 'Gold Dragon', 'Guard Leader', 'Guardian', \
        'Guardian #2', "Hell's Rider", 'Hidon', 'Holy Dragon', 'Humbaba', 'Humbaba #2', 'Humbaba #3', 'Humbaba #4', \
        'Ice Dragon', 'Ifrit', 'Inferno', 'Ipooh', 'Kaiser Dragon', 'Kefka (Narshe)', 'Kefka (Final)', \
        'Ketu', 'Lady', 'Laragorn', 'Laser Gun', 'Left Blade', 'Leviathan', 'Long Arm', 'Machine', \
        'Magic', 'Magic Master', 'Magitek Armor', 'Malboro Menace', 'Master Tonberry', 'Missile Bay', \
        'Moebius', 'Nelapa', 'Neslug (Head)', 'Neslug (Shell)', 'Number 024', 'Number 128', \
        'Omega Weapon', 'Phantom Train', 'Plague', 'Power', 'Rahu', 'Red Dragon', 'Rest', 'Rhizopas', \
        'Right Blade', 'Samurai Soul', 'Shiva', 'Short Arm', 'Siegfried', 'Siegfried #2', 'Skull Dragon', 'Soul Saver', \
        'Storm Dragon', 'Tentacle (Bottom-Right)', 'Tentacle (Bottom-Left)', 'Tentacle (Top-Right)', \
        'Tentacle (Top-Left)', 'Tiger', 'Tonberries', 'Tunnel Armor', 'Typhon', 'Typhon #2', \
        'Ultima Buster', 'Ultima Weapon', 'Ultros', 'Ultros #2', 'Ultros #3', 'Ultros #4', \
        'Valigarmanda', 'Valigarmanda #2', 'Vargas', 'Visage', 'Wrexsoul', 'Yeti', 'Ymir (Head)', 'Ymir (Shell)']

    df['Boss'] = [1 if enemy in bosses else 0 for enemy in df.index]
    return df


def add_ios_col(df):
    ios_names = [idx for idx in df.index if len(df.loc[idx, 'SNES_Name']) <3]
    df['iOS'] = [1 if name in ios_names else 0 for name in df.index]
    return df


def add_dragon_den_col(df):
    den_bosses = ['Earth Eater', 'Gargantua', 'Malboro Menace', 'Dark Behemoth', 'Abyss Worm', 'Neslug (Head)', 'Neslug (Shell)', 'Plague', 'Flan Princess', 'Ice Dragon #2', 'Storm Dragon #2', 'Earth Dragon #2', 'Blue Dragon #2', 'Red Dragon #2', 'Skull Dragon #2', 'Holy Dragon #2', 'Gold Dragon #2', 'Kaiser Dragon', 'Omega Weapon']
    den_enemies = ['Dinozombie', 'Zurvan', 'Vilia', 'Great Dragon', 'Magic Dragon', 'Armodullahan', 'Crystal Dragon', 'Maximera', 'Death Rider', 'Abaddon', 'Shield Dragon', 'Dragon Aevis', 'Hexadragon']
    df['Dragon_Den'] = [1 if mob in den_bosses + den_enemies else 0 for mob in df.index]
    return df


def make_snes_df(df):
    df_snes = df[df['iOS'] == 0]
    df_snes = df_snes.set_index('SNES_Name')
    return df_snes


def plot_scatter(df, x_axis='Lvl', y_axis='HP', mask_var='Boss'):
    x = df[x_axis].values
    y = df[y_axis].values

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    mask = df[mask_var] == 1
    ax.scatter(x[mask], y[mask], s=115, alpha=.8, color='FireBrick', label=mask_var)
    ax.scatter(x[~mask], y[~mask], s=105, alpha=.7, color='DodgerBlue', label='Normal')
    # plt.grid()
    for idx, name in enumerate(df.index.tolist()):
        ax.annotate(name, (x[idx], y[idx]), fontsize=8)

    plt.title("Enemies of Final Fantasy VI")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend(loc='best')

    plt.show()
    plt.close()


def prep_data(target='Boss'):
    df_mod = df_raw.select_dtypes(include=['int64', 'float64'])
    y = df_mod.pop(target).values
    X = df_mod.loc[:, df_mod.columns != target].values

    class_ratio = Counter(y)
    print("{t} classes have {zp}:{op} ratio. 0: {zero}, 1: {one}".format(t=target.capitalize(), zp=int(100*class_ratio.get(0)/len(y)), op=int(100*class_ratio.get(1)/len(y)), zero=class_ratio.get(0), one=class_ratio.get(1)))

    return X, y, df_mod


def split_data(X, y, df, train_size=.70, random_state=None):
    # class_ratio = Counter(y)
    # if .40 < class_ratio.most_common(1)[0][1] / sum(class_ratio.values()) < .60:
    #     stratify = y
    return train_test_split(X, y, df, train_size=train_size, stratify=y)


def make_model(X_train, y_train, class_weight=None, random_state=None):
    mod = RandomForestClassifier(n_estimators=500, max_depth=12, max_leaf_nodes=6, class_weight=class_weight, random_state=random_state)
    mod.fit(X_train, y_train)
    return mod


def report_mean_results(X, y, df_mod, n=10, train_size=.7, p_thresh=.0, random_state=None):
    print("Fitting {n} models to find mean importances and probabilities...".format(n=n))

    def update_proba_dict(X, proba_dict):
        probas = mod.predict_proba(X)[:, 1]
        for name in df_train.index:
            if not probas[df_train.index.get_loc(name)]:
                print("----------", name)
            proba_dict[name].append(probas[df_train.index.get_loc(name)])
        return proba_dict

    importances, train_scores, test_scores = [], [], []
    proba_dict = defaultdict(list, [])

    for run in range(n):
        print("Run:", run+1)
        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df_mod, train_size=train_size, random_state=random_state, stratify=y)
        mod = make_model(X_train, y_train, class_weight='balanced_subsample', random_state=random_state)

        importances.append(mod.feature_importances_)
        proba_dict = update_proba_dict(X_train, proba_dict)
        train_scores.append(mod.score(X_train, y_train))
        test_scores.append(mod.score(X_test, y_test))
        report_scores(mod, X_train, X_test, y_train, y_test)

    importances = np.mean(importances, axis=0)
    report_importances(importances, df_mod)

    for name, probas in proba_dict.items():
        proba_dict[name] = np.mean(probas)

    probabilities = pd.DataFrame.from_dict(dict(proba_dict), orient='index')
    probabilities.columns = ['proba_mean']
    mean_frame = report_probabilities(df_raw, probabilities.sort_index(), target, p_thresh=p_thresh)

    train_scores = np.mean(train_scores)
    test_scores = np.mean(test_scores)
    print("Train Accuracy: {a:.2f}%".format(a=train_scores*100))
    print("Test Accuracy: {a:.2f}%\n".format(a=test_scores*100))



    return mean_frame


def report_importances(importances, df):
    print("\nThe {n} most important features were:".format(n=len(importances)))
    # df_train_mod = df_train.drop('Boss', axis=1).select_dtypes(include=['int64', 'float64'])
    imps = [(feat, imp) for feat, imp in zip(df.columns, importances)]
    imps = sorted(imps, key=lambda x: x[1], reverse=True)
    for idx, (feat, imp) in enumerate(imps, 1):
        print("{idx}. {feat}: {imp:.2f}%".format(idx=idx, feat=feat, imp=imp*100))


def report_probabilities(df, probabilities, target, p_thresh=.0, above_below='above'):
    """Pass in matching X and df splits, such as X_train and df_train.  Use p_thresh = 0 to return all entries in df. above_below controls whether the threshold for probability should be above or below set point."""
    if isinstance(probabilities, pd.DataFrame):
        pred_targ_names = probabilities.index.values
        probabilities = probabilities['proba_mean'].tolist()
    else:
        pred_targ_names = df_train.index.values
        sort_frame = pd.DataFrame(index=pred_targ_names, data={'proba_mean': probabilities}).sort_index()
        probabilities = sort_frame['proba_mean'].values

    names_mask = df.index.isin(pred_targ_names)
    df.loc[names_mask, 'Proba_'+target] = probabilities
    df.loc[~names_mask, 'Proba_'+target] = np.nan
    df.loc[names_mask, 'Pred_'+target] = [target if x > .5 else 'No' for x in df.loc[names_mask, 'Proba_'+target]]
    df.loc[names_mask, 'Right_Pred?'] = ['Yes' if (a == 1 and b > .5) or (a == 0 and b <=.5) else 'No' for a, b in zip(df.loc[names_mask, 'Boss'], df.loc[names_mask, 'Proba_'+target])]

    thresh_mask = df.loc[:, 'Proba_'+target] >= p_thresh if above_below == 'above' else df.loc[:, 'Proba_'+target] <= p_thresh
    df_thresh = df[thresh_mask]

    # report_stats = ['SNES_Name', 'Attack', 'Defense', 'Evasion', 'Speed', 'Boss', 'Proba_'+target, 'Pred_'+target, 'Right_Pred?']
    report_stats = ['SNES_Name', 'Boss', 'Proba_'+target, 'Pred_'+target, 'Right_Pred?']

    print(df_thresh[report_stats].sort_values('Proba_'+target, ascending=False))
    return df[report_stats].sort_values('Proba_'+target, ascending=False)


def report_scores(mod, X_train, X_test, y_train, y_test):
    print("\nTrain Accuracy: {a:.2f}%".format(a=mod.score(X_train, y_train)*100))
    print("Test Accuracy: {a:.2f}%".format(a=mod.score(X_test, y_test)*100))





if __name__ == '__main__':
    monster_dict = load_text('data/FF6_bestiary.txt')
    df_raw = make_df(monster_dict)
    df_snes = make_snes_df(df_raw)
    plot_scatter(df_snes, 'Level', 'HP', 'Boss')
    target = 'Boss'
    # X, y, df_mod = prep_data(target)

    # mean_frame = report_mean_results(X, y, df_mod, n=10, train_size=.7, p_thresh=.75, random_state=None)
    #
    # X_train, X_test, y_train, y_test, df_train, df_test = split_data(X, y, df_mod, train_size=.7, random_state=462)
    # mod = make_model(X_train, y_train, random_state=462)
    #
    # report_importances(mod.feature_importances_, df_train)
    # mainn = report_probabilities(df_raw, mod.predict_proba(X_train)[:, 1], target, p_thresh=.75)
    # report_scores(mod, X_train, X_test, y_train, y_test)
