#!/bin/sh
set -eu

if [ -f /tmp/.X99-lock ]; then
	old_pid=$(cat /tmp/.X99-lock 2>/dev/null || true)
	case "$old_pid" in
		''|*[!0-9]*) ;;
		*) kill "$old_pid" 2>/dev/null || true ;;
	esac
	rm -f /tmp/.X99-lock
fi
rm -f /tmp/.X11-unix/X99

Xvfb :99 -screen 0 1280x900x24 -ac &
until xdpyinfo -display :99 >/dev/null 2>&1; do
	sleep 1
done
fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display :99 -forever -shared -rfbport 5900 -nopw >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/websockify.log 2>&1 &

exec python arima_forecast.py --data AirPassengers.csv