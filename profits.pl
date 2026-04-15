#!/usr/bin/perl
use strict;
use URI::Escape;
use HTML::Entities;
use HTTP::Request;
use LWP::UserAgent;
use LWP::Parallel::UserAgent;
use JSON;
use Data::Dumper;
use DateTime::Format::Epoch;
use Tie::IxHash;
use Number::Format 'format_number';
use List::Util qw/sum/;

sub mean {
    return sum(@_)/@_;
}

my $json = JSON->new->allow_nonref;

my $ua = LWP::UserAgent->new;
$ua->max_redirect(5);
$ua->agent("Mozilla/5.0 (Windows NT 6.1)");

my $formatter = DateTime::Format::Epoch->new(
                    epoch          => DateTime->new( year => 1970, month => 1, day => 1 ),
                    unit           => 'seconds',
                    type           => 'int',    # or 'float', 'bigint'
                    skip_leap_seconds => 1,
                    start_at       => 0,
                    local_epoch    => undef,
                );

# potions
my @items = qw/37971 37969 37939 55324 55326 49115 48626 49117/;
# ingredients
foreach my $i (qw/23191 37973 34159 12181 37227 55697 25501 25503 48962 48921/) {
  push @items, $i;
}
#elder rune
foreach my $i (qw/44830 44832 2363 44844/) {
  push @items, $i;
}

my %items;

foreach my $itemid (@items) {
  my $req = HTTP::Request->new(GET => "http://services.runescape.com/m=itemdb_rs/api/catalogue/detail.json?item=$itemid");

  my $iName;
  my $res = $ua->request($req);
  if ($res->is_success) {
    my $data = $json->decode( $res->content );
    $iName =$data->{item}->{name};
    $items{$data->{item}->{name}}->{id} = $data->{item}->{id};
    $items{$data->{item}->{name}}->{members} = $data->{item}->{members};
  } else {
    print $res->status_line, "\n";
    print $res->content;
  }

  my (@daily, @average, @dts, @ats, @elVol, %nfo);

  $req = HTTP::Request->new(GET => "http://services.runescape.com/m=itemdb_rs/api/graph/$itemid.json");
  my $res = $ua->request($req);
  if ($res->is_success) {
    my $data = $json->decode( $res->content );
    #print Dumper(\$data);
    foreach my $ms (sort {$b <=> $a} keys %{$data->{daily}}) {
#      if ($#daily < 10) {
        my $day = split(/T/,$formatter->parse_datetime($ms/1000))[0];
        $nfo{$day}->{daily} = $data->{daily}->{$ms};
#      }
    }
    foreach my $ms (sort {$b <=> $a} keys %{$data->{average}}) {
#      if ($#average < 10) {
        my $day = split(/T/,$formatter->parse_datetime($ms/1000))[0];
        $nfo{$day}->{daily} = $data->{average}->{$ms};
#      }
    }

  } else {
    print $res->status_line, "\n";
    print $res->content;
  }

  $req = HTTP::Request->new(GET => "https://api.weirdgloop.org/exchange/history/rs/last90d?id=$itemid");
  my $res = $ua->request($req);
  if ($res->is_success) {
    my $data = $json->decode( $res->content );
    foreach my $el (reverse @{$data->{$itemid}}) {
        my $ts = $formatter->parse_datetime($el->{timestamp}/1000);
        my $day = split(/T/,$ts)[0];
        $nfo{$day}->{vts} = $ts;
        $nfo{$day}->{vprice} = $el->{price};
        $nfo{$day}->{volume} = $el->{volume};
      push @elVol, $el;
    }
  } else {
    print $res->status_line, "\n";
    print $res->content;
  }

  #print "daily prices len ".$#daily." vol90 len ".$#elVol."\n";

  foreach my $key (keys %nfo) {
  #  printf("ts: %s\tprice: %u\tavg price: %u\tvolume: %u\tvol ts:%s\n", 
  #
      push @{$items{$iName}->{ts}}, sprintf("%s", $formatter->parse_datetime($dts[$i]));
      push @{$items{$iName}->{daily}}, $nfo{$key}->{daily}[$i];
      push @{$items{$iName}->{average}}, $average[$i];
    #just incase dates dont line up or volume is missing data
    if ($i <= $#elVol) {
      if ($daily[$i] == $elVol[$i]->{price}) {
        push @{$items{$iName}->{volume}}, $elVol[$i]->{volume};
        push @{$items{$iName}->{vts}}, sprintf("%s",$formatter->parse_datetime($elVol[$i]->{timestamp}/1000));
      } else {

        warn sprintf("price mismatch i: %u, %u != %u", $i, $daily[$i], $elVol[$i]->{price});
      }
    }
  #  );
  }
}

#print Dumper(\%items);
#exit;

my $amount = 2400;
my $costFirst = $items{'Bloodweed potion (unf)'}->{daily}[0];
my $costSecond = $items{'Searing ashes'}->{daily}[0];
my $costFlask = $items{'Potion flask'}->{daily}[0];
my $price3dose = $items{'Aggression potion (3)'}->{daily}[0];
my $price4dose = $items{'Aggression potion (4)'}->{daily}[0];
my $price6dose = $items{'Aggression flask (6)'}->{daily}[0];

my $cost = $costSecond * $amount / 1.111 + $costFirst * $amount;
my $doses = ($amount * 0.85 * 1.15 * 3 + $amount * 0.15 * 1.15 * 4);
my $g3 = $doses / 3 * $price3dose;
my $g4 = $doses / 4 * $price4dose;
my $g6 = $doses / 6 * $price6dose;
my $net3 = ($doses / 3 * $price3dose) - $cost;
my $net4 = ($doses / 4 * $price4dose) - $cost;
my $net6 = ($doses / 6 * $price6dose) - ($cost + $costFlask * ($doses/6));

printf "Aggression potions, amount $amount\nBloodweed potions(unf) %s Searing ashes %s Flask %s total %s\n", $costFirst, $costSecond, $costFlask, format_number($cost);
printf "latest price 3dose %s 4dose %s flask %s\n", $price3dose, $price4dose, $price6dose;
printf "amount $amount 3dose %s 4dose %s flask %s\n", format_number($net3), format_number($net4), format_number($net6);

printf "10 day average volume 3dose %s 4dose %s flask %s\n\n", 
  mean(@{$items{'Aggression potion (3)'}->{volume}}[0,10]),
  mean(@{$items{'Aggression potion (4)'}->{volume}}[0,10]),
  mean(@{$items{'Aggression flask (6)'}->{volume}}[0,10]);

# extreme necromancy

printf "Extreme Necromancy IMPLEMENT ME\n";
my $amount = 2700;
printf "\nWeapon poison: amount %s\n", format_number($amount);
my @in = ('Weapon poison++ (3)', 'Poison slime', 'Primal extract');
my @out = ('Weapon poison+++ (3)', 'Weapon poison+++ (4)', 'Weapon poison+++ flask (6)');
my $costFlask = $items{'Potion flask'}->{daily}[0];
my $price3dose = $items{$out[0]}->{daily}[0];
my $price4dose = $items{$out[1]}->{daily}[0];
my $price6dose = $items{$out[2]}->{daily}[0];
my $costFirst = $items{$in[0]}->{daily}[0];
my $costSecond = $items{$in[1]}->{daily}[0];
my $costThird = $items{$in[2]}->{daily}[0];

my $cost = $costThird * $amount / 1.111 + $costSecond * $amount / 1.111 + $costFirst * $amount;
my $doses = ($amount * 0.85 * 1.15 * 3 + $amount * 0.15 * 1.15 * 4);
my $dosesMax = ($amount * 0.75 * 1.15 * 3 + $amount * 0.25 * 1.15 * 4);
my $g3 = $doses / 3 * $price3dose;
my $g4 = $doses / 4 * $price4dose;
my $g6 = $doses / 6 * $price6dose;
my $net3 = ($doses / 3 * $price3dose) - $cost;
my $net4 = ($doses / 4 * $price4dose) - $cost;
my $net6 = ($doses / 6 * $price6dose) - ($cost + $costFlask * ($doses/6));
my $net3Max = ($dosesMax / 3 * $price3dose) - $cost;
my $net4Max = ($dosesMax / 4 * $price4dose) - $cost;
my $net6Max = ($dosesMax / 6 * $price6dose) - ($cost + $costFlask * ($dosesMax/6));

printf "latest price 3dose %s 4dose %s flask %s\n", $price3dose, $price4dose, $price6dose;
printf "amount $amount net profit: 3dose %s 4dose %s flask %s\n", format_number($net3), format_number($net4), format_number($net6);
printf "amount $amount max possible net profit: 3dose %s 4dose %s flask %s\n", format_number($net3Max), format_number($net4Max), format_number($net6Max);

printf "10 day average volume 3dose %s 4dose %s flask %s\n", 
  map { mean(@{$items{$_}->{volume}}[0, 10]) } @out;

# weapon poison
my $amount = 2400;
printf "\nWeapon poison: amount %s\n", format_number($amount);
my @in = ('Weapon poison++ (3)', 'Poison slime', 'Primal extract');
my @out = ('Weapon poison+++ (3)', 'Weapon poison+++ (4)', 'Weapon poison+++ flask (6)');
my $costFlask = $items{'Potion flask'}->{daily}[0];
my $price3dose = $items{$out[0]}->{daily}[0];
my $price4dose = $items{$out[1]}->{daily}[0];
my $price6dose = $items{$out[2]}->{daily}[0];
my $costFirst = $items{$in[0]}->{daily}[0];
my $costSecond = $items{$in[1]}->{daily}[0];
my $costThird = $items{$in[2]}->{daily}[0];

my $cost = $costThird * $amount / 1.111 + $costSecond * $amount / 1.111 + $costFirst * $amount;
my $doses = ($amount * 0.85 * 1.15 * 3 + $amount * 0.15 * 1.15 * 4);
my $dosesMax = ($amount * 0.75 * 1.15 * 3 + $amount * 0.25 * 1.15 * 4);
my $g3 = $doses / 3 * $price3dose;
my $g4 = $doses / 4 * $price4dose;
my $g6 = $doses / 6 * $price6dose;
my $net3 = ($doses / 3 * $price3dose) - $cost;
my $net4 = ($doses / 4 * $price4dose) - $cost;
my $net6 = ($doses / 6 * $price6dose) - ($cost + $costFlask * ($doses/6));
my $net3Max = ($dosesMax / 3 * $price3dose) - $cost;
my $net4Max = ($dosesMax / 4 * $price4dose) - $cost;
my $net6Max = ($dosesMax / 6 * $price6dose) - ($cost + $costFlask * ($dosesMax/6));

printf "latest price 3dose %s 4dose %s flask %s\n", $price3dose, $price4dose, $price6dose;
printf "amount $amount net profit: 3dose %s 4dose %s flask %s\n", format_number($net3), format_number($net4), format_number($net6);
printf "amount $amount max possible net profit: 3dose %s 4dose %s flask %s\n", format_number($net3Max), format_number($net4Max), format_number($net6Max);

printf "10 day average volume 3dose %s 4dose %s flask %s\n", 
  map { mean(@{$items{$_}->{volume}}[0, 10]) } @out;

# incense sticks
#
printf "\nIncense sticks:  the acadia ones, lantadyme, etc... IMPLEMENT ME!\n\n";

# smelting rune bars
my @amounts = (2800,10000,30000);
my ($in1, $in2, $in3) = ('Rune bar', 'Light animica', 'Dark animica');
my $out1 = 'Elder rune bar';
my $sale_price = $items{$out1}->{daily}[0];
my $ucost = $items{$in1}->{daily}[0] + $items{$in2}->{daily}[0] + $items{$in3}->{daily}[0];
printf "$out1: amount per hour %s, per 4 hour buy limit %s, per night %s\n",
  map { format_number($_) } @amounts;

foreach my $amount (@amounts) {
  my $cost = $ucost * $amount;
  my $gross = $sale_price * $amount * 1.12;
  my $grossMax = $sale_price * $amount * 1.14;
  printf "hours %0.2f amount %s: cost %s net %s net max %s\n", 
    $amount/$amounts[0], format_number($amount), format_number($cost), format_number($gross - $cost), format_number($grossMax - $cost);
}

my %overrides;
do {
  print "Apply overrides?\n";
  my $input = <>;
  chomp($input);
  if (exists $items{$input}) {
    $overrides{$input} = $input;
  }
} while (keys %overrides > -1);
