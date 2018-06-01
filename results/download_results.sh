if [ ! -d "favos_baseline" ]; then
    wget https://www.dropbox.com/s/9zwob31bz91u75h/favos.tar
    tar -xf favos.tar
    rm -f favos.tar
fi

