cd Movie_Recommendations

wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
mv ml-100k/ua.base data/raw
mv ml-100k/ua.test data/raw
mv ml-100k/u.user data/raw
mv ml-100k/u.genre data/raw
mv ml-100k/u.item data/raw

cd data/interim
gdown https://drive.google.com/uc?id=1VLrWudGAGcQDRHiKDx-T72XfAmU5FUdY
gdown https://drive.google.com/uc?id=19GAIyyOM1TYTWmMa99XFHCOpwcaFLKGB
gdown https://drive.google.com/uc?id=1n5CWF21lXCy9Sx8Mh5-eGe3PEKBfz0dL
gdown https://drive.google.com/uc?id=1g3SHTa_lNFtP8YtM7jrSGdfeau4RfrsL

cd ../../benchmark/data
gdown https://drive.google.com/uc?id=1UYSvHSjFj__xB0CJrskIdfW0bSZXwut9

cd ..
python benchmarkevaluate.py
