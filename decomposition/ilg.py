

class SspgNode():
    def __init__(self, text, uid):
        self.text = text
        self.uid = uid
    
    def as_string(self):
        return f"{self.text}_{self.uid}"

class Sspg():
    def __init__(self, source, outgoing, join_incoming=None):
        self.source = source
        self.outgoing = outgoing
        self.join_incoming = None
        # If self.outgoing == [] then this is a sink node.
        # If len(self.outgoing) == 1 then the outgoing edge is a seq edge (equiv to sync)
        # Otherwise, self.outgoing[0] is the sync-child and other children are ordered-split children.

    def children(self):
        return [e.v for e in self.outgoing]

    def as_string_list(self, indent_increment=2, join_successor=None):
        # Right now, this prints it like a tree.
        if len(self.outgoing) == 0:
            if self.join_incoming is None:
                return []
            else:
                return f"{self.source.as_string()} -> {self.join_incoming.u.as_string()}"
        
        str_list = [self.outgoing[0].as_string()]
        for e in self.outgoing[1:]:
            e_str_list = e.as_string_list(indent_increment=indent_increment)
            e_str_list = [f"{' '*indent_increment}{e_str}" for e_str in e_str_list]
            str_list.extend(e_str_list)
        

        

        


class SspgEdge():
    def __init__(self, u, v, text):
        self.u = u
        self.v = v
        self.text = text

    def as_string(self):
        return f"{self.u.as_string()} -> {self.v.as_string}"