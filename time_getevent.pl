#!/usr/bin/perl

$SIG{INT} = sub { 

  close SH; 
  open(SH, "<", "record.sh") || die $!;
  my @out;
  @out = <SH>;
  close SH;
  pop @out;
  print @out;
  open(SH, ">", "record.sh") || die $!;
  print SH @out;
  print SH "sleep 1\n";
  print SH "done\n";
  close SH;

};

#todo: record time from DOWN to UP

#parse adb shell wm size
my ($maxX, $maxY) = (1200,1920);

my $fh;
open(SH, ">", "record.sh") || die $!;
print SH '#!/bin/sh'."\n";
print SH 'while [ 1 ]; do'."\n";
open($fh, "-|", 'adb shell getevent -tl') || die $!;

my $orientation = "LANDSCAPE";
my $event = undef;
my $lastEvent = undef;
my $firstEvent = undef;
my $r;
while(<$fh>) {
    chomp;
    my ($null, $time, $dev,$ev,$type,$val) = split(/\s+/,$_);
    if ($ev eq 'EV_ABS') {
        if ($type eq 'ABS_MT_TOUCH_MAJOR') {
        }
        if ($type =~ /^ABS_MT_POSITION/) {
            $val = hex($val);
            if ($orientation eq 'LANDSCAPE') {
                #        printf("%s\t%s\n",$type, $val);
                if ($type =~ /X$/) {
                    $event->{'y'} = $maxX - $val;
                }
                if ($type =~ /Y$/) {
                    $event->{'x'} = $val;
                }
            }
        }
        if ($type =~ /TRACKING_ID$/) {
            #            printf("id %s\tx,y (%i,%i)\n",$val,$event->{'x'}, $event->{'y'});
            #            printf("adb shell input swipe %i %i %i %i 100\n",$event->{'x'}, $event->{'y'}, $event->{'x'}, $event->{'y'});
        }
        #printf("%s\t%s\n",$type, $val);
    } elsif ($ev eq 'EV_KEY') {
        if ($val eq 'DOWN') {
            $lastEvent = $event;
            if (not defined $firstEvent) {
              $firstEvent = $event;
            }
            $event = {};
            $r = '';
            $time =~ s/\]//;
            $event->{start} = $time * 1000;
        } elsif ($val eq 'UP') {
            $time =~ s/\]//;
            $event->{end} = $time * 1000;
            $event->{time} = ($event->{end} - $event->{start});
            printf "%d %d %d\n", $event->{start}, $event->{end}, $event->{time};
            if (defined $lastEvent) {
              my $diff = ($event->{end} - $lastEvent->{end}) / 1000;
              printf SH "sleep %0.1f\n", $diff;
              printf("sleep %0.1f\n", $diff);
            } else {
              printf SH <<EOF
if ($SECONDS -gt 0:              
            }
            printf SH "adb shell input swipe %i %i %i %i %d\n",$event->{'x'}, $event->{'y'}, $event->{'x'}, $event->{'y'}, $event->{time};
            printf("adb shell input swipe %i %i %i %i %d\n",$event->{'x'}, $event->{'y'}, $event->{'x'}, $event->{'y'}, $event->{time});
        }
    }
}
close SH;


