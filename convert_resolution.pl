#!/usr/bin/perl
#
#
use strict;
use Getopt::Long;
use Data::Dumper;

#resolution 2560x1600
#phone #  resolution 2400x1080

my ($file, $outfile, $resolution, $tag) = ('','','0x0', '');
GetOptions("file=s" => \$file,
            "resolution=s" => \$resolution,
            "tag=s" => \$tag
            );

my ($fromW, $fromH, $w, $h, $scaleW, $scaleH);

  open(SH, "<", $file) || die $!;
  my @in;
  @in = <SH>;
  close SH;

  for my $l (@in) {
    if ($l =~ /resolution (\d+)x(\d+)/) {
      $fromW = $1;
      $fromH = $2;
    }
  }
if ($resolution =~ /(\d+)x(\d+)/) {
  $w = $1;
  $h = $2;
}

$scaleW = 1/($fromW / $w);
$scaleH = 1/($fromH / $h);

print "input file resolution ($fromW, $fromH)\n";
print "output resolution ($w, $h)\n";
print "scale factor ($scaleW, $scaleH)\n";

if ($file =~ /(.*?)(\..*)/) {
  my ($pfx, $ext) = ($1, $2);
  if (not defined $tag) {
    $tag = sprintf("%sx%s",$w,$h);
  }

  $outfile = sprintf("%s_%s%s",$pfx,$tag,$ext);
} else {
  $outfile = "converted$file";
}

my @out;
for my $l (@in) {
  my $line = $l;
  if ($l =~ /(.*?)adb shell input swipe (\d+) (\d+) (\d+) (\d+) (\d+)(.*)/s) {
    my ($pre,$x,$y,$x2,$y2,$t,$post) = ($1,$2,$3,$4,$5,$6,$7);
    $x=int($2*$scaleW);
    $x2=int($4*$scaleW);
    $y=int($3*$scaleH);
    $y2=int($5*$scaleH);
    $line = sprintf("%sadb shell input swipe %s %s %s %s %s%s", $pre,$x,$y,$x2,$y2,$t,$post);
  }
  push @out, $line;
}
open(OUT, ">$outfile") || die $!;
print OUT @out;
close(OUT);
print "written $outfile\n";
