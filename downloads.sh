googleimagesdownload --keywords "Temple Mount, Pyramids of Giza, Chichen Itza, Moai Statues, The Colosseum, The Great Wall of China, Angkor Wat, Petra, Taj Mahal, Machu Pichu" --limit 10

mkdir -p ml_data/test

mv downloads/* ml_data/test

googleimagesdownload --keywords "Temple Mount, Pyramids of Giza, Chichen Itza, Moai Statues, The Colosseum, The Great Wall of China, Angkor Wat, Petra, Taj Mahal, Machu Pichu" --of 11 --limit 50

mkdir -p ml_data/train

mv downloads/* ml_data/train

tar czf ml.tar.gz ml_data