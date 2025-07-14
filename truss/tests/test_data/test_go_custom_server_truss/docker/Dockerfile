FROM golang:1.23-bookworm

WORKDIR /home
COPY main.go /home/main.go
RUN go build -o server main.go
ENTRYPOINT ["./server"]
