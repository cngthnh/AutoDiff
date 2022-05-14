from .graph import Graph

_graphs = {}

class Session():

    """
    This class is used to manage the graph.
    """

    _id = -1
    _defaultId = -1

    def __init__(self):
        Session._id += 1
        self.id = Session._id
        Session._defaultId = self.id
        global _graphs
        _graphs[self.id] = Graph()
    
    @staticmethod
    def getGraph():
        global _graphs
        try:
            return _graphs[Session._defaultId]
        except KeyError:
            return None

    @staticmethod
    def setDefault(sess):
        Session._defaultId = sess.id

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
    
    def reset(self):
        global _graphs
        _graphs[self.id].clear()

    def close(self):
        global _graphs
        try:
            del _graphs[self.id]
            if Session._defaultId == self.id:
                keys = list(_graphs.keys())
                if (len(keys) < 1):
                    Session._defaultId = -1
                else:
                    Session._defaultId = keys[-1]
        except:
            pass