#!/usr/bin/perl

use DBM::Deep;
use Data::Dumper;

my $filename = shift || 'sleepycat.db';
my $field = shift;
my $db =  tie %hash, 'DBM::Deep',
            {file => $filename}
    or die "Cannot open file $filename\n" ;

print Dumper(\%hash);
