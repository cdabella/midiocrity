#!/bin/bash
if test -f "clean_midi.tar.gz"; then
    echo "clean_midi.tar.gz already present. Skipping download"
else
    curl -c cookies "https://drive.google.com/uc?export=download&id=1Unl3MjqBg0YaQQSso4tH2dFq7NrhK2hA" > tmp.html
    curl -L -b cookies "https://drive.google.com$(cat tmp.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > clean_midi.tar.gz
    rm tmp.html
    rm cookies
fi

if md5sum --status -c check_clean_midi.md5; then
    echo "clean_midi.tar.gz MD5 correct"
else 
    echo "clean_midi.tar.gz file corrupted. Redownload file"
    exit 1
fi

echo "Extracting clean_midi"
tar -xzvf clean_midi.tar.gz
rm clean_midi.tar.gz