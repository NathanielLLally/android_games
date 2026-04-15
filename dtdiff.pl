use strict;
use warnings;
use DateTime;

my $dt2 = DateTime->new(
                       year   => 2010,
                       month  => 8,
                       day    => 2,
                       hour   => 9,
                       minute => 10,
                       second => 8,
                       time_zone => 'local',
                     );

my $dt1 = DateTime->now( time_zone => 'local' )->set_time_zone('floating');

my $dur = $dt1->subtract_datetime($dt2);
print 'hours = ', $dur->hours(), "\n";
print 'hours = ', $dur->in_units('hours'), "\n";

my $dur2 = DateTime::Duration->new( years => 24, months => 15 );
printf "%s %s %s %s\n", $dur2->in_units('years', 'days', 'hours');

