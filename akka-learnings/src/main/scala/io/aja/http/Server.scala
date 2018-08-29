package io.aja.http

import java.time.LocalDateTime

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer

import scala.concurrent.{Future, blocking}
import scala.io.StdIn

//Messages
case class ss()

object Server {
  //https://www.gregbeech.com/2018/04/08/akka-http-client-pooling-and-parallelism/
  implicit val system: ActorSystem = ActorSystem("http-slow-server")
  implicit val mat: ActorMaterializer = ActorMaterializer()
  import system.dispatcher

  def main(args: Array[String]): Unit = {


    def slowOp(requestUri: Uri): Future[Unit] = Future {
      blocking {
        println(s"[${LocalDateTime.now}] --> ${requestUri.authority.host}")
        Thread.sleep(1000)
      }
    }

    //https://doc.akka.io/docs/akka-http/current/client-side/request-level.html
    val route = extractUri { uri =>
      onSuccess(slowOp(uri)) {
        complete(StatusCodes.NoContent)
      }
    }

    val binding: Future[Http.ServerBinding] =
      Http().bindAndHandle(route, "0.0.0.0", 4000)

    println("Server online at http://0.0.0.0:4000/")

    StdIn.readLine()

    binding.flatMap(_.unbind()).onComplete(_ => system.terminate())
  }

}
