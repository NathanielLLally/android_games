#!/usr/bin/perl
use locale;
my $o = sprintf("%\'d\n", 12345667);
print $o;

my %h;
$h{'test(5)'} = 5;

print $h{'test(5)'};
my $key = 'test(5)';
print $h{$key};

my @a = (0,1,2);
print $#a."\n";
