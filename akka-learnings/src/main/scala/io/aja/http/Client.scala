package io.aja.http


import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import scala.util.{Failure, Success}
import java.time.LocalDateTime

object Client {
  implicit val system: ActorSystem = ActorSystem("http-pool-test")
  implicit val mat: ActorMaterializer = ActorMaterializer()
  import system.dispatcher

  def main(args: Array[String]): Unit =
    for (i <- 1 to 64) {

//      val request = HttpRequest(uri = Uri("http://localhost:4000/"))

      val request = HttpRequest(uri = Uri(s"http://localhost${i % 2 + 1}:4000/"))

      Http().singleRequest(request).onComplete {
        case Success(_) => println(s"[${LocalDateTime.now}] $i succeeded")
        case Failure(e) => println(s"[${LocalDateTime.now}] $i failed: $e")
      }

    }
}
