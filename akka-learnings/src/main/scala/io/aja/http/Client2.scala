package io.aja.http

import java.time.LocalDateTime

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{Sink, Source}

import scala.util.{Failure, Success}

object Client2 {
  implicit val system: ActorSystem = ActorSystem("http-pool-test")
  implicit val mat: ActorMaterializer = ActorMaterializer()
  import system.dispatcher

  val connection1 = Http().cachedHostConnectionPool[Int]("localhost", 4000)
  val connection2 = Http().cachedHostConnectionPool[Int]("localhost", 4000)

  def main(args: Array[String]): Unit = {
    Source(1 to 32)
      .map(i => (HttpRequest(uri = Uri("http://localhost:4000/")), i))
      .via(connection1)
      .runWith(Sink.foreach {
        case (Success(_), i) => println(s"[${LocalDateTime.now}] $i succeeded")
        case (Failure(e), i) => println(s"[${LocalDateTime.now}] $i failed: $e")
      })
    Source(33 to 64)
      .map(i => (HttpRequest(uri = Uri("http://localhost:4000/")), i))
      .via(connection2)
      .runWith(Sink.foreach {
        case (Success(_), i) => println(s"[${LocalDateTime.now}] $i succeeded")
        case (Failure(e), i) => println(s"[${LocalDateTime.now}] $i failed: $e")
      })
  }
}
