import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
from sklearn.manifold import TSNE
from sklearn import decomposition
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import cytoolz


class ElmoEmbedder:
    """ Takes a structured dataset with columns that contain text data needing to be embedded
    
        Usage:
            1) init class with your data, chosen list of names for text columns, and an option final
                embedding size (e.g. 50, 100, 250 et)
            2) call the make_embeddings method
            3) optionally save the elmo embeddings for the sentences currently in state
            4) call the reduce_dims method if you'd like to use a reduce feature set of the 
                original ELMo embeddings
            5) optionally visualise the semantic embeddings for each row by running visualise_embeddings
            6) you can now access the embeddings you need by calling get_embeddings_for_training
            
    """
    def __init__(self, data, text_cols, final_embedding_size=None, use_gpu=False):
        """ Takes in data and processes it to 'sentences' made of each text column entry comma delineated
            Args:
                data: pandas dataframe containing data for training
                text_cols: a list of column names that map to the data columns containing unprocessed text
                final_embedding_size: size of the output ELMo dimension reduced sentence embedding
                use_gpu: boolean - currently doesn't quite work, but will be a future feature
        
        """
        self.embed_size = final_embedding_size
        
        sentences = [] # this will be where we add all the row entries per column as 'sentences'
        for index, row in data.iterrows():
            curr_sent = ""
            for c in range(len(text_cols)):
                col = text_cols[c]
                entry = row[col]
                entry = entry.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ') #get rid of problem chars
                entry = ' '.join(entry.split()) #a quick way of removing excess whitespace
                entry = self.process(entry)
                curr_sent += entry
                if c != len(text_cols) - 1:
                    curr_sent += ', '
            sentences.append(curr_sent)
        
        self.sentences = sentences
        self.use_gpu = use_gpu
    
    def process(self, text):

        text = text.replace("\n", " ")

        #     for char in ['!','?',',', ':', ';', '/','(',')', '\\', '-', ']', '[']:
        #         text = text.replace(char, "")

        words = text.split(' ')
        for i in range(len(words)):
            w = words[i]
            try:
                float(w)
            except:
                words[i] = words[i].replace('.', "")
        text = " ".join(words)
        text = text.lower()
        return text

    def make_embeddings(self, load=False, filename=None):
        """ Embed all the sentences as ELMo embeddings
        
            Args:
                load: if True, load from file using the given filename
                filename: string of filename for saved ELMo embeddings, if None,
                    loader defaults to 'elmo_embeddings.npy' to load from
        
        """
        if load:
            self.load_elmo_embeddings(filename)
            return

        # Get the ELMo model
        url = "https://tfhub.dev/google/elmo/2"
        embed_model = hub.Module(url)

        all_embeddings = []
#         if self.use_gpu:
#             device = '/gpu:0'
#         else:
#             device = '/cpu:0'
#         with tf.device(device):
        for sentence_block in cytoolz.partition_all(150, self.sentences):
            embeddings = embed_model(
                sentence_block,
                signature="default",
                as_dict=True)["default"]
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                x = sess.run(embeddings)
            all_embeddings.extend(x)
        
        self.elmo_embeddings = np.array(all_embeddings)
        self.reduced_embeddings = self.elmo_embeddings
    
    def reduce_dims(self, size=None):
        if size == None:
            size = self.embed_size
        
        try:
            pca = decomposition.PCA(n_components=size)
            self.reduced_embeddings = pca.fit_transform(self.elmo_embeddings)
        except AttributeError as e:
            print("Oops, looks like the embeddings haven't been generated yet! \
                Try calling elmo_embedder.make_embeddings() first!")
    
    def visualise_embeddings(self):
        y = TSNE(n_components=2).fit_transform(self.reduced_embeddings) # further reduce to 2 dim using t-SNE
        init_notebook_mode(connected=True)
        data = [
            go.Scatter(
                x=[i[0] for i in y],
                y=[i[1] for i in y],
                mode='markers',
                text=[i for i in self.sentences],
            marker=dict(
                size=16,
                color = [len(i) for i in self.sentences], #set color equal to a variable
                opacity= 0.8,
                colorscale='Viridis',
                showscale=False
            )
            )
        ]
        layout = go.Layout()
        layout = dict(
                      yaxis = dict(zeroline = False),
                      xaxis = dict(zeroline = False)
                     )
        fig = go.Figure(data=data, layout=layout)
        file = plot(fig, filename='sentence_encode.html')
    
    def save_elmo_embeddings(self, filename=None):
        if filename == None:
            filename = 'elmo_embeddings.npy'
        
        np.save(filename, self.elmo_embeddings)
    
    def load_elmo_embeddings(self, filename):
        if filename == None:
            filename = 'elmo_embeddings.npy'
        
        self.elmo_embeddings = np.load(filename)
        self.reduced_embeddings = self.elmo_embeddings
    
    def get_embeddings_for_training(self):
        return self.reduced_embeddings

