if [ ! -d "DAVIS2016" ]; then
    wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
    unzip DAVIS-data.zip
    mv DAVIS DAVIS2016
    rm -f DAVIS-data.zip
fi
