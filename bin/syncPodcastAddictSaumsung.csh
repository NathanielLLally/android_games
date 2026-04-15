#!/bin/csh -f
#  -f prevents loading .cshrc, providing more uniform execution

GUEST_CACHE_DIR="/storage/emulated/0/Android/media/com.bambuna.podcastaddict"
#"/storage/emulated/0/Android/data/com.spotify.music/files/spotifycache/Storage"
HOST_CACHE_DIR="/tmp/podcast_addict/"

if ( -e "$HOST_CACHE_DIR" ) then
  echo "todo remove existing host cache"
else 
  mkdir -p $HOST_CACHE_DIR
endif 


echo "cleaning mp4 from guest dir"
CMD="adb shell find $GUEST_CACHE_DIR -iname '*.mp4' -exec rm '{}' \;"
`$CMD`;

# no rsync on droid
#  -g, --listed-incremental=FILE  ?

echo "streaming tarball"
cd $HOST_CACHE_DIR
adb shell "cd $GUEST_CACHE_DIR && tar -czv . " -- | tar -xzv 

echo ""
mkdir -p "$HOST_CACHE_DIR/appDump"
#unzip "$GUEST_CACHE_DIR/" -d "$HOST_CACHE_DIR/appDump" 
#cleanup
echo rm -rf $HOST_CACHE_DIR

