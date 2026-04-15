#!/usr/bin/perl
use strict;
use URI::Escape;
use HTML::Entities;
use HTTP::Request;
use LWP::Parallel::UserAgent;
use JSON;
use Data::Dumper;
use DateTime::Format::Epoch;
use Tie::IxHash;

my $json = JSON->new->allow_nonref;

my $ua = LWP::Parallel::UserAgent->new;
$ua->redirect(1);
$ua->duplicates(0);
$ua->max_req(4);
$ua->max_hosts(20);
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
my @items = qw/37971 37969 37939 55324 55326 49115 48626/;
# ingredients
foreach my $i (qw/37973 34159 12181 37227 55697 25501 25503 48962 48921/) {
  push @items, $i;
}
#elder rune
foreach my $i (qw/44830 44832 2363 44844/) {
  push @items, $i;
}

my %items;

sub on_return_detail() {
  my ($request, $response, $entry) = @_;
  my $data = $json->decode( $response->content );
    print Dumper(\$data);
  printf("id: %u name: %s member: %s\n",
      $data->{item}->{id}, $data->{item}->{name}, $data->{item}->{members}
      );
}

sub on_return_graph() {
  my ($request, $response, $entry) = @_;
  my $data = $json->decode( $response->content );
    print Dumper(\$data);

  my (@daily, @average, @dts, @ats);
    foreach my $ms (sort {$b <=> $a} keys %{$data->{daily}}) {
      if ($#daily < 10) {
        push @daily, $data->{daily}->{$ms};
        push @dts, $ms/1000;
      }
    }
    foreach my $ms (sort {$b <=> $a} keys %{$data->{average}}) {
      if ($#average < 10) {
        push @average, $data->{average}->{$ms};
        push @ats, $ms/1000;
      }
    }
}

sub on_return_volume() {
  my @elVol;
  my ($request, $response, $entry) = @_;
    my $data = $json->decode( $response->content );
    print Dumper(\$data);

    #foreach my $el (reverse @{$data->{$itemid}}) {
    #  push @elVol, $el;
    #}
}
foreach my $itemid (@items) {
  my $req = HTTP::Request->new(GET => "http://services.runescape.com/m=itemdb_rs/api/catalogue/detail.json?item=$itemid");
  $ua->register($req, \&on_return_detail);

  $req = HTTP::Request->new(GET => "http://services.runescape.com/m=itemdb_rs/api/graph/$itemid.json");
  $ua->register($req, \&on_return_graph);

  $req = HTTP::Request->new(GET => "https://api.weirdgloop.org/exchange/history/rs/last90d?id=$itemid");
  $ua->register($req, \&on_return_volume);

=head

  foreach my $i (0..9) {
    #just incase dates dont line up or volume is missing data
    if ($daily[$i] != $elVol[$i]->{price}) {
      die sprintf("price mismatch i: %u, %u != %u", $i, $daily[$i], $elVol[$i]);
    }
    printf("ts: %s\tprice: %u\tavg price: %u\tvolume: %u\tvol ts:%s\n", 
      $formatter->parse_datetime($dts[$i]), $daily[$i], $average[$i],
      $elVol[$i]->{volume}, $formatter->parse_datetime($elVol[$i]->{timestamp}/1000)
    );
  }
=cut

}

my $entries = $ua->wait(50);
print Dumper(\$entries);
exit;

my $costFirst = shift;
my $costSecond = shift;
my $price4dose = shift;
my $amount = shift || 2500;

#print "using flask?";
#my $flask = <STDIN>;
#chomp $flask;

my $cost = $costSecond * $amount / 1.111 + $costFirst * $amount;
my $gross = ($amount * 0.85 * 1.15 * 3 + $amount * 0.15 * 1.15 * 4) / 4 * $price4dose;
my $net = $gross - $cost;
print "cost $cost gross $gross net $net\n";


