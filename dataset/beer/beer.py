from ..sequence import WordSequence

class Beer(object):

    def __init__(self,
                 id,
                 name,
                 brewer,
                 style,
                 abv):
        self.id = id
        self.name = name
        self.brewer = brewer
        self.style = style
        self.abv = abv

    @staticmethod
    def from_json(obj):
        return Beer(
            int(obj['beer/beerId']),
            obj['beer/name'],
            int(obj['beer/brewerId']),
            obj['beer/style'],
            obj.get('beer/ABV', None)
        )

class Review(object):

    def __init__(self,
                 beer,
                 rating,
                 text,
                 user
                 ):
        self.beer = beer
        self.rating = rating
        self.text = text
        self.user = user

    @staticmethod
    def from_json(obj, beer):
        return Review(
            beer,
            obj['review/overall'],
            obj['review/text'],
            obj['user/profileName'],
        )
