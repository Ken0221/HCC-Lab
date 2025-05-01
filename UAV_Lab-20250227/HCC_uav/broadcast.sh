#!/bin/bash
for ip in 170.20.10.{1..254}; do
  sudo arp -d $ip > /dev/null 2>&1
  ping -c 5 $ip > /dev/null 2>&1 &
done
wait
arp -na | grep -v incomplete
