# PINNeAPPle Playground: a static site (the models run in the browser). Build from the repository root:
#   docker build -f apps/playground/Dockerfile -t pinneapple-playground .
#   docker run -p 8095:8095 pinneapple-playground
# Stage 1 assembles the site with the standard library only; stage 2 serves it with Caddy.
FROM python:3.11-slim AS build
WORKDIR /src
COPY apps/playground apps/playground
COPY pinneapple_hub/zoo/web pinneapple_hub/zoo/web
COPY pinneapple_tools/visualization/studio/web/vendor/three pinneapple_tools/visualization/studio/web/vendor/three
COPY data/geometries data/geometries
COPY data/web data/web
RUN python apps/playground/build.py --out /site

FROM caddy:2.8.4-alpine
COPY apps/playground/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /site /srv
EXPOSE 8095
HEALTHCHECK --start-period=10s CMD wget -q -O /dev/null http://localhost:8095/index.html || exit 1
