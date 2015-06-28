class CornellMoviesDataset(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.movies = {}
        with open(self.data_dir / 'movie_titles_metadata.txt') as fp:
            for line in fp:
                movie = Movie.from_line(line)
                self.movies[movie.id] = movie

        self.lines = {}
        with open(self.data_dir / 'movie_lines.txt') as fp:
            for line in fp:
                line = Line.from_line(line, self.movies)
                self.lines[line.id] = line

        self.conversations = []
        with open(self.data_dir / 'movie_conversations.txt') as fp:
            for line in fp:
                conversation = Conversation.from_line(line, self.lines)
                self.conversations.append(conversation)

class Movie(object):

    def __init__(self, id, title, year, rating, genres):
        self.id = id
        self.title = title
        self.year = year
        self.rating = rating
        self.genres = genres

    @staticmethod
    def from_line(line):
        line = [m.strip() for m in line.strip().split('+++$+++')]
        assert len(line) == 6, "Improperly formatted movie: %s" % str(line)
        movie = Movie(line[0], line[1], int(line[2][:4]), float(line[3]), line[5])
        return movie

    def __repr__(self):
        return "Movie<%s (%u)>" % (self.title, self.year)

class Line(object):

    def __init__(self, id, movie, character, text):
        self.id = id
        self.movie = movie
        self.character = character
        self.text = text

    @staticmethod
    def from_line(line, movies):
        line = [m.strip() for m in line.strip().split('+++$+++')]
        assert len(line) == 5, "Improperly formatted line: %s" % str(line)
        line = Line(line[0], movies[line[2]], line[3], line[4])
        return line

class Conversation(object):

    def __init__(self, lines, movie):
        self.lines = lines
        self.movie = movie
        self.length = len(self.lines)

    @staticmethod
    def from_line(line, lines):
        line = [m.strip() for m in line.strip().split('+++$+++')]
        assert len(line) == 4, "Improperly formatted conversation: %s" % str(line)
        lines = [lines[l] for l in eval(line[3])]
        return Conversation(lines, lines[0].movie)

    def __str__(self):
        return "\n".join(
            ["%s: %s" % (l.character, l.text) for l in self.lines]
        )
