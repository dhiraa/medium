package io.aja

import akka.actor.SupervisorStrategy.{Escalate, Restart}
import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, OneForOneStrategy, PoisonPill, Props, SupervisorStrategy}

import scala.util.{Failure, Success}
import akka.pattern._
import akka.routing.{DefaultResizer, RoundRobinPool}
import akka.util.Timeout

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.concurrent.duration._


//========================================================================================

//Messages
case class Message(text: String)
case class Ping()
case class Pong()
case class GetDetails(id: Long)
case class InvokeChildActor(msg: String)
case class KillChildActor()

case class TestRoundRobin()

//========================================================================================

case class WorkerFailedException(error: String) extends Exception(error)

//========================================================================================
//http://allaboutscala.com/scala-frameworks/akka/
class ChildActor extends Actor with ActorLogging {

  override def preStart(): Unit = log.info("prestart")

  override def postStop(): Unit = log.info("postStop")

  override def preRestart(reason: Throwable, message: Option[Any]): Unit = log.info("preRestart")

  override def postRestart(reason: Throwable): Unit = log.info(s"restarting ${self.path.name} because of $reason")

  override def receive: Receive = {
    case Message(text) =>
      log.info("Child Actor -> " + text)
    case KillChildActor =>
//       throw new IllegalStateException("boom") // Will Escalate the exception up the hierarchy
       throw new WorkerFailedException("boom") // Will Restart DonutStockWorkerActor

      context.stop(self) // For the sake of this example, after procssing the CheckStock message, the
    // ChildActor will stop processing any other messages as we're stopping it by calling context.stop(self).
    case TestRoundRobin =>
      log.info(s"thread = ${Thread.currentThread().getId}")
  }
}

//========================================================================================

class SimpleActor extends Actor with ActorLogging {

  override def supervisorStrategy: SupervisorStrategy =
    OneForOneStrategy(
      maxNrOfRetries = 3,
      withinTimeRange = 1 second,
      loggingEnabled = true) {
      case _: WorkerFailedException =>
        log.error("Worker failed exception, will restart.")
        Restart

      case _: Exception =>
        log.error("Worker failed, will need to escalate up the hierarchy")
        Escalate
    }

  val childActor = context.actorOf(Props[ChildActor], "ChildActor")

  // We are using a resizable RoundRobinPool.
  val resizer = DefaultResizer(lowerBound = 5, upperBound = 10)
  val props = RoundRobinPool(nrOfInstances = 5,
    resizer = Some(resizer), supervisorStrategy = supervisorStrategy)
    .props(Props[ChildActor])
  val roundRobinChildActorRouterPool: ActorRef = context.actorOf(props, "RoundRobinRouter")

  def mockDetailsLookup(id: Long): Future[String] = Future {
    "Dummy"
  }

  override def preStart(): Unit = log.info("prestart")

  override def postStop(): Unit = log.info("postStop")

  override def preRestart(reason: Throwable, message: Option[Any]): Unit = log.info("preRestart")

  override def postRestart(reason: Throwable): Unit = log.info("postRestart")

  override def receive: Receive = {
    case Message(text) =>
      log.info("Received msg : " + text)
    case Ping =>
      log.info("Actor -> Ping")
      sender ! Pong
    case Pong =>
      log.info("Actor -> Pong")
      sender ! Ping
    case GetDetails(id) =>
      log.info("Actor -> GetDetails")
      mockDetailsLookup(id).pipeTo(sender)
    case InvokeChildActor(msg) =>
      log.info("Actor -> InvokeChildActor")
      childActor ! Message(msg)
    case KillChildActor =>
      log.info("Actor => KillChildActor")
      childActor ! KillChildActor
    case TestRoundRobin =>
      roundRobinChildActorRouterPool ! TestRoundRobin
  }
}

//========================================================================================

object FirstExample {

  def main(args: Array[String]): Unit = {

    implicit val timeout = Timeout(5 second)

    val system = ActorSystem("FirstExample")


    val simpleActor = system.actorOf(Props[SimpleActor], name = "SimpleActor")

    simpleActor ! Message("hey there...")  //Tell pattern

    val response1 = simpleActor ? Ping    //Ask pattern
    response1.foreach(res => println("Response -> "  + res))

    val response2 = simpleActor ? Pong //Ask pattern
    response2.foreach(res => println("Response -> "  + res))

    val response3 = simpleActor ? GetDetails(123) //PipeTo pattern
    response3.foreach(res => println("Response -> "  + res))

    system.actorSelection("/user/SimpleActor") ! Message("wow this also works?")

    simpleActor ! InvokeChildActor("lets try this")

    val response4 = simpleActor ? KillChildActor
    response4.foreach(res => println("Response -> "  + res))

    val response5 = (0 to 25).map(_ => simpleActor ? TestRoundRobin)
    response5.foreach(res => println("Response -> "  + res))

    Thread.sleep(2000)

    simpleActor ! PoisonPill
    simpleActor ! Message("After Poison ;)")  //Tell pattern

    val isTerminated = system.terminate()


    isTerminated.onComplete{
      case Success(result) => println("Success " + result)
      case Failure(error) => println(error)
    }

    Thread.sleep(2000)

  }
}

//========================================================================================
