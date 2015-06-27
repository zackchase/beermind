#!/bin/sh
#curl http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip > data/movies.zip
unzip data/movies.zip -d data/
mv "data/cornell movie-dialogs corpus/" data/corpus/
rm -rf data/__MACOSX
