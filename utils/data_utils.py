import os
import gensim
import numpy as np
import collections
import pretty_midi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def get_metadata(base_path):
    """
        :param base_path:
        :outputs df_metadata, df_num_music_piece:
    """
    composer_names = os.listdir(base_path)
    metadata = collections.defaultdict(list)
    map_composer_number_music_piece = {}
    composer_data = {}
    for composer_name in composer_names:
        compositions_dir = os.path.join(base_path, composer_name)
        compositions = os.listdir(compositions_dir)
        map_composer_number_music_piece[composer_name] = len(compositions)
        sub_dict = {}
        for comp in compositions:
            title = comp.split("_")[0]
            midi_filename = os.path.join(composer_name,comp)
            metadata["composer"].append(composer_name)
            metadata["title"].append(title)
            metadata["midi_filename"].append(midi_filename)
            sub_dict[title] = sub_dict.get(title, 0)+1
        composer_data[composer_name] = sub_dict
    df_metadata = pd.DataFrame(metadata)

    composer_number_music_piece = {k:v for k,v in sorted(map_composer_number_music_piece.items(), key=lambda x: x[1])}
    df_num_music_piece = pd.DataFrame({"Composers": composer_number_music_piece.keys(), "Number of music pieces":composer_number_music_piece.values()})
    return df_metadata, df_num_music_piece, composer_data




# Plot functions

def plot_composer_dist(df_num_music_piece):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    sns.barplot(data = df_num_music_piece, y="Composers", x="Number of music pieces", orient='h', ax=ax1)
    colors = sns.color_palette()[0:len(df_num_music_piece)]
    data = df_num_music_piece["Number of music pieces"].values
    labels= df_num_music_piece["Composers"].values
    #create pie chart
    ax2.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    plt.show()
    fig.tight_layout(pad=7.0)



def plot_freq_composition(composer_data):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
    plt.subplots_adjust(hspace=4)
    fig.suptitle("Composer-wise Composition Frequencies", fontsize=13, y=0.95)
    
    for (composer, compositions), ax in zip(composer_data.items(), axs.ravel()):
        compositions = dict(sorted(compositions.items(), key=lambda x: x[1], reverse=True))
        compositions_names = list(compositions.keys())
        compositions_freq = list(compositions.values())
        ax.bar(compositions_names, compositions_freq, color='skyblue')
        ax.set_xlabel('Compositions')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{composer}')
        ax.tick_params(axis='x', labelrotation = 90)
        ax.set_ylim([0,7])
    fig.tight_layout()
    plt.show()


def convert_midifiles_to_notes(midi_file: str) -> pd.DataFrame:
    """
        :param midi_file:
        :output pd.DataFrame:
    """
    notes = collections.defaultdict(list)
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        notes['velocity'].append(note.velocity)
        prev_start = start
    df_notes = pd.DataFrame({k: np.array(v) for k, v in notes.items()})
    return df_notes


def extract_gram(midi_frame: pd.DataFrame):
    """
        :param midi_file:
        :output :
    """
    gram_list = []
    temp = []
    s_time = 0
    for i in range(midi_frame.shape[0]):
        pitch = midi_frame["pitch"][i]
        ti = round(midi_frame["duration"][i],2)
        gram = (pitch,ti)

    if((not temp) or (midi_frame["start"][i] - s_time <= 0.003)):
        temp.append(gram)
        if(len(temp) == 1):
            s_time = midi_frame["start"][i]
        if(i == midi_frame.shape[0] - 1):
            gram_list += temp
    else:
        sorted_list = sorted(temp, key=lambda tup: tup[0], reverse=True)
        sorted_list.append(gram)
        gram_list += sorted_list
        temp.clear()
        s_time = 0

    return gram_list


def encodeChinese(index_number):
    """
        :param index_number:
        :output :
    """
    val = index_number + 0x4e00
    return chr(val)


def get_sentence_vector_avg(sentences,model):
    """
        :param sentences:
        :param model:
        :output :
    """
    l = []
    for sentence in sentences:
        for word in sentence:
            try:
                temp = np.zeros(len(model.wv[word]))
                temp += model.wv[word]
            except:
                print("Not in vocab")
    l.append(temp/len(sentence))
    return l



def get_sentence_vector_avg_and_sd(sentences,model):
    """
        :param sentences:
        :param model:
        :output :
    """
    l = []
    cov = []
    for sentence in sentences:
        for word in sentence:
            try:
                temp = np.zeros(len(model.wv[word]))
                temp += model.wv[word]
                cov.append(model.wv[word])
            except:
                print("Not in vocab")
        data = np.array(cov)
        sd = np.std(data,axis=0)
        z = temp/len(sentence)
        z = z.tolist()
        z += sd.tolist()
        z = np.array(z)
        l.append(z)
    return l



def get_sentence_vector_sd(sentences,model):
    """
        :param sentences:
        :param model:
        :output :
    """
    l = []
    cov = []
    for sentence in sentences:
        for word in sentence:
            try:
                cov.append(model.wv[word])
            except:
                print("Not in vocab")
        data = np.array(cov)
        sd = np.std(data,axis=0)
        z = sd.tolist()
        z = np.array(z)
        l.append(z)
    return l


def create_label(sentences_list, df_metadata):
    """
        :param sentences_list:
        :param df_metadata:
        :output :
    """
    df = df_metadata.groupby(["composer"])["title"].count().to_frame()
    df.reset_index(inplace=True)
    composer_map = { j:i for i,j in enumerate(df["composer"])}
    for i in df_metadata.index.tolist():
        try:
            data.append(sentences_list[i])
            label.append(composer_map[df_metadata.iloc[i]["composer"]])
        except:
            print("Error",i)
    return data, label
     

def create_label(sentences_list, df_metadata):
    """
        :param sentences_list:
        :param df_metadata:
        :output data, label:
    """
    data = []
    label = []
    df = df_metadata.groupby(["composer"])["title"].count().to_frame()
    df.reset_index(inplace=True)
    composer_map = { j:i for i,j in enumerate(df["composer"])}
    for i in df_metadata.index.tolist():
        try:
            data.append(sentences_list[i])
            label.append(composer_map[df_metadata.iloc[i]["composer"]])
        except:
            print("Error",i)
    return data, label, composer_map