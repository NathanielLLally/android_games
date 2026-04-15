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
use DateTime::Format::ISO8601;
use Date::Manip;
use DateTime::Format::DateManip;
use Tie::IxHash;
use Number::Format 'format_number';
use List::Util qw/sum/;
use Text::Table;
use Try::Tiny;
use DBM::Deep;

sub mean {
    return int(sum(@_)/@_+0.5);
}

my %items;
my $filename = "ge_item_nfo.db";
my $db = tie %items, "DBM::Deep", {
    file      => $filename,
    locking   => 1,
    autoflush => 1
} or die "Cannot open $filename";

my $updateDb = undef;
if (exists $items{updated}) {
  print "last updated: ". $items{updated},"\n";
  my $dt = DateTime::Format::DateManip->parse_datetime(ParseDate($items{updated}));
  my $yesterday = DateTime->now()->subtract(hours => 12);
  print "yesterday: ". DateTime::Format::ISO8601->format_datetime($yesterday);
  print "\n".DateTime->compare($yesterday, $dt) ."\n";
  if ( DateTime->compare($yesterday, $dt) > 0) {
    $updateDb = 1;
  }
} else {
  $updateDb = 1;
}

if (defined $updateDb) {
    print "prices out of date, refreshing\n";
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
foreach my $i (qw/23191 37973 34159 37227 55697 25501 25503 48962 48921 12181 55697/) {
  push @items, $i;
}
#48962
#pb of sorc
foreach my $i (qw/43979 43975 43977 43981 14856 44055 43989 44079 48925 565 49063/) {
  push @items, $i;
}
#incense, kwuarm (acadia, accrused, grimy and clean herb, incense)
foreach my $i (qw/40285 20266 213 263 47709 1515 20268 2485 2481 2481 47713 12174 47705 12172 20264 1519/) {
  push @items, $i;
}

#elder rune
foreach my $i (qw/44830 44832 2363 44844/) {
  push @items, $i;
}


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
        push @daily, $data->{daily}->{$ms};
        push @dts, $ms/1000;
#      }
    }
    foreach my $ms (sort {$b <=> $a} keys %{$data->{average}}) {
#      if ($#average < 10) {
        push @average, $data->{average}->{$ms};
        push @ats, $ms/1000;
#      }
    }

  } else {
    print $res->status_line, "\n";
    print $res->content;
  }

  $req = HTTP::Request->new(GET => "https://api.weirdgloop.org/exchange/history/rs/last90d?id=$itemid");
  my $res = $ua->request($req);
  if ($res->is_success) {
    try {
      my $data = $json->decode( $res->content );
      foreach my $el (reverse @{$data->{$itemid}}) {
        push @elVol, $el;
      }
    }catch {
    };
  } else {
    print "$itemid\n";
    print $res->status_line, "\n";
    print $res->content;
  }

  #print "daily prices len ".$#daily." vol90 len ".$#elVol."\n";

  print "$iName\n";

  $items{$iName}->{ts} = ();
  $items{$iName}->{daily} = ();
  $items{$iName}->{average} = ();
  $items{$iName}->{volume} = ();
  $items{$iName}->{vts} = ();

  foreach my $i (0..$#daily) {
  #  printf("ts: %s\tprice: %u\tavg price: %u\tvolume: %u\tvol ts:%s\n", 
  #
      push @{$items{$iName}->{ts}}, sprintf("%s", $formatter->parse_datetime($dts[$i]));
      push @{$items{$iName}->{daily}}, $daily[$i];
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

$items{updated}= DateTime::Format::ISO8601->format_datetime(DateTime->now());
}

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

my $amount = 2850;
printf "\nExtreme necromancy: pots/hr %s\n", format_number($amount);
my @in = ('Spirit weed potion (unf)', 'Congealed blood', 'Ground miasma rune');
my @out = ('Extreme necromancy (3)', 'Extreme necromancy (4)');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;


sub  calcNetSplitAmt {
  my $sa = shift;
  my $sb = $amount - $sa; 
  my $cost = 5 * $priceIn[1] * $sa / 1.111 + $priceIn[0] * $sa + $priceIn[2] * $sb / 1.111;

  my $doses = ($sb * 0.85 * 1.15 * 3 + $sb * 0.15 * 1.15 * 4);
  my $net3 = ($doses / 3 * $priceOut[0]) - $cost;
  my $net4 = ($doses / 4 * $priceOut[1]) - $cost;
  printf "Amount made $sb net profit: 3dose %s 4dose %s\n", format_number($net3), format_number($net4);
}

printf "Costs %s\t%s\t%s\n", @in;
printf "\t%s\t%s\t%s\n", @priceIn;
printf "latest price 3dose %s 4dose %s\n", @priceOut;

calcNetSplitAmt(int(($amount / 2)/1.15+0.5));
print "max ";
calcNetSplitAmt(int(($amount / 2)/1.2074+0.5));

printf "10 day average volume 3dose %s 4dose %s\n", 
  map { mean(@{$items{$_}->{volume}}[0, 10]) } @out;

# super runecrafting / pb of sorc

my $amount = 1000;
printf "\nPowerburst of sorcery:\n";
my @in = ('Runecrafting potion (3)', 'Yak milk', 'Primal extract', 'Blood rune', 'Beak snot');
#my @out = ('Super runecrafting (3)');
my @out = ('Powerburst of sorcery (4)');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;

my @amounts = ($amount, $amount/1.111,$amount*1.2074,($amount*3*1.2074)/1.111,($amount*1.2074)/1.111);
printf "purchase amounts, $amount ingredients: %s,\t%s,\t%s,\t%s,\t%s\n\t\t%s\t%s\t%s\t%s\t%s\n",
       @in, map { format_number(int($_+0.5)); } @amounts;

#amount per hour
$amount = 2500;
sub pbCalcNetSplitAmt {
  my $sa = shift;
  my $sb = $amount - $sa;
  my $cost = $priceIn[1] * $sa / 1.111 + $priceIn[0] * $sa;
  $cost += $priceIn[2] * $sb + 3*$priceIn[3] * $sb / 1.111 + $priceIn[4]*$sb/1.111;
    # potions
  #my $doses = ($sb * 0.85 * 1.15 * 3 + $sb * 0.15 * 1.15 * 4);
  #my $net3 = ($doses / 3 * $priceOut[0]) - $cost;
  #
  my $net = $priceOut[0] * int($sb * 1.15+0.5) - $cost;

my @amounts = ($sa, $sa/1.111,$sa*1.2074,($sa*3*1.2074)/1.111,($sa*1.2074)/1.111);
printf "  $sa ingredients: %s,\t%s,\t%s,\t%s,\t%s\n\t\t%s\t%s\t%s\t%s\t%s\n",
       @in, map { format_number(int($_+0.5)); } @amounts;

  printf "pots/hr %s yields %s Powerburst of sorcery, price %s total cost %s net profit %s\n", 
    map { format_number($_); } ($amount, int($sb*1.15+0.5), $priceOut[0], $cost, $net);
}

#use all available boosts for super runecrafting
pbCalcNetSplitAmt(int($amount*0.453+0.5));

my @l = ('Super runecrafting (3)', @out);
printf "volumes for %s\t%s\n10 day avg\t%s\t%s\npast 24 h\t%s\t%s\n",
  @l, map { format_number(mean(@{$items{$_}->{volume}}[0, 10])); } @l,
    map { format_number(@{$items{$_}->{volume}}[0]) } @l;

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
my $amount = 675;
my @in = ('Acadia logs', 'Accursed ashes', 'Grimy kwuarm');
my @out = ('Kwuarm incense sticks');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;

my $cost = $priceIn[0] * $amount * 2 + $priceIn[1]*$amount*2 + $priceIn[2];
my $net = ($amount * $priceOut[0]) - $cost;
printf "\n\n%s\n", @out;
printf "current prices %s %s\n", @out, map { format_number($_); } @priceOut;
printf "\t%s %s %s: %s %s %s\n", @in, map { format_number($_); } @priceIn;
printf "amount $amount cost %s net profit: %s\n", map { format_number($_); } ($cost, $net);
printf "10 day average volume %s: %s\n", @out, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @out;

my $amount = 675;
my @in = ('Yew logs', 'Infernal ashes', 'Grimy lantadyme');
my @out = ('Lantadyme incense sticks');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;

my $cost = $priceIn[0] * $amount * 2 + $priceIn[1]*$amount*2 + $priceIn[2];
my $net = ($amount * $priceOut[0]) - $cost;
printf "\n\n%s\n", @out;
printf "current prices %s %s\n", @out, map { format_number($_); } @priceOut;
printf "\t%s %s %s: %s %s %s\n", @in, map { format_number($_); } @priceIn;
printf "amount $amount cost %s net profit: %s\n", map { format_number($_); } ($cost, $net);
printf "10 day average volume %s: %s\n", @out, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @out;

my $amount = 675;
my @in = ('Willow logs', 'Impious ashes', 'Clean spirit weed');
my @out = ('Spirit weed incense sticks');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;

my $cost = $priceIn[0] * $amount * 2 + $priceIn[1]*$amount*2 + $priceIn[2];
my $net = ($amount * $priceOut[0]) - $cost;
printf "\n\n%s\n", @out;
printf "current prices %s %s\n", @out, map { format_number($_); } @priceOut;
printf "\t%s %s %s: %s %s %s\n", @in, map { format_number($_); } @priceIn;
printf "amount $amount cost %s net profit: %s\n", map { format_number($_); } ($cost, $net);
printf "10 day average volume %s: %s\n", @out, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @out;

#cleaning herbs
#
my $amount = 60000;
my @in = ('Grimy lantadyme');
my @out = ('Clean lantadyme');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;

my $cost = $priceIn[0] * $amount;
my $gross = ($amount * $priceOut[0]);
printf "\n\n%s\n", @out;
printf "current prices %s %s\n", @out, map { format_number($_); } @priceOut;
printf "\t%s %s\n", @in, map { format_number($_); } @priceIn;
printf "amount $amount cost %s gross %s net profit: %s\n", map { format_number($_); } ($cost, $gross, $gross - $cost);
printf "\n10 day average volume %s,%s,%s %s\n", @in, @out, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @in, @out;

my $amount = 60000;
my @in = ('Grimy kwuarm');
my @out = ('Clean kwuarm');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;
my $cost = $priceIn[0] * $amount;
my $gross = ($amount * $priceOut[0]);
printf "\n\n%s\n", @out;
printf "current prices %s %s\n", @out, map { format_number($_); } @priceOut;
printf "\t%s %s\n", @in, map { format_number($_); } @priceIn;
printf "amount $amount cost %s gross %s net profit: %s\n", map { format_number($_); } ($cost, $gross, $gross - $cost);
printf "\n10 day average volume %s,%s,%s %s\n", @in, @out, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @in, @out;

my $amount = 60000;
my @in = ('Grimy spirit weed');
my @out = ('Clean spirit weed');
my @priceIn = map { $items{$_}->{daily}[0] } @in;
my @priceOut = map { $items{$_}->{daily}[0] } @out;
my $cost = $priceIn[0] * $amount;
my $gross = ($amount * $priceOut[0]);
printf "\n\n%s\n", @out;
printf "current prices %s %s\n", @out, map { format_number($_); } @priceOut;
printf "\t%s %s\n", @in, map { format_number($_); } @priceIn;
printf "amount $amount cost %s gross %s net profit: %s\n", map { format_number($_); } ($cost, $gross, $gross - $cost);
printf "\n10 day average volume %s,%s,%s %s\n", @in, @out, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @in, @out;

#
# smelting rune bars
my @amounts = (2800,10000,30000);
my @in = ('Rune bar', 'Light animica', 'Dark animica');
my @out = 'Elder rune bar';
my $sale_price = $items{$out[0]}->{daily}[0];
my $ucost = $items{$in[0]}->{daily}[0] + $items{$in[1]}->{daily}[0] + $items{$in[2]}->{daily}[0];
printf "\n\n".$out[0].": amount per hour %s, per 4 hour buy limit %s, per night %s\n",
  map { format_number($_) } @amounts;

foreach my $amount (@amounts) {
  my $cost = $ucost * $amount;
  my $gross = $sale_price * $amount * 1.12;
  my $grossMax = $sale_price * $amount * 1.14;
  printf "hours %0.2f amount %s: cost %s net %s net max %s\n", 
    $amount/$amounts[0], format_number($amount), format_number($cost), format_number($gross - $cost), format_number($grossMax - $cost);
}

printf "10 day average volume %s: %s\n", @out, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @out;

printf "prev 3 days volume %s,%s,%s\n", @in;
printf "latest\t%s %s %s\n", map { format_number(@{$items{$_}->{volume}}[0]) } @in;
printf "t-1\t%s %s %s\n", map { format_number(@{$items{$_}->{volume}}[1]) } @in;
printf "t-2\t%s %s %s\n", map { format_number(@{$items{$_}->{volume}}[2]) } @in;

printf "\n10 day average volume %s,%s,%s %s %s %s\n", @in, 
  map { format_number(mean(@{$items{$_}->{volume}}[0, 10])) } @in;

my %overrides;
do {
  print "Apply overrides?\n";
  my $input = <>;
  chomp($input);
  if (exists $items{$input}) {
    $overrides{$input} = $input;
  }
} while (keys %overrides > -1);
