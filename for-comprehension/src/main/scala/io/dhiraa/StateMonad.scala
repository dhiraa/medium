package io.dhiraa

/*
First, I doubt that anyone wrote a State
monad like this on their first try. I’m sure it took several efforts before someone
figured out how to get what they wanted in a for expression.

Second, while this code can be hard to understand in one sitting, I’ve looked at some
of the source code inside the Scala collections classes, and there’s code in there
that’s also hard to grok. (Take a look at the sorting algorithms and you’ll see what I
mean.) Personally, the only way I can understand complex code like this is to put it
in a Scala IDE and then modify it until I make it my own.
 */

case class StateWithoutDebugInfo[S, A](run: S => (S, A)) {

  def flatMap[B](g: A => StateWithoutDebugInfo[S, B]): StateWithoutDebugInfo[S, B] = StateWithoutDebugInfo { (s0: S) =>
    val (s1, a) = run(s0)
    g(a).run(s1)
  }

  def map[B](f: A => B): StateWithoutDebugInfo[S, B] = flatMap(a => StateWithoutDebugInfo.point(f(a)))
}

object StateWithoutDebugInfo {
  def point[S, A](v: A): StateWithoutDebugInfo[S, A] = StateWithoutDebugInfo(run = s => (s, v))
}



/**
  * A class that maintains th state of type S and aggregates them over the time as a type B from A
  * @param run
  * @tparam S StateM Type
  * @tparam A Aggregated Type
  */
case class StateM[S, A](run: S => (S, A)) {

  //As a naive way of remembering, flat map always has to return a new type in the wrapper.
  //Here the state remain the same, A -> B
  def flatMap[B](g: A => StateM[S, B]): StateM[S, B] = StateM { (currentState: S) =>
    println("\n>>> flatmap: currentState " + currentState + " calling run/swing on current state...")
    val (nextState, a) = run(currentState)
//    val res = g(a).run(nextState)
//    println("<<< aggregated state info " + res)
//    res
    println("--- flatmap: nextState " + nextState + " new distance " + a + ". calling g...")
    val stateChangeToB = g(a)
    println("--- flatmap: actually triggering run/swing on next state after initializing g()...\n")
    val res = stateChangeToB.run(nextState)
    println("<<< flatmap: aggregated state info " + res)
    res
  }

  /**
    * Map function that does apply some logic to convert the type A -> B. Eg: 10 -> 10+5 or 10 -> 10,
    * while keeping the state the same. I think this must have come after try
    * @param f Function that applies the logic on flatmapped value/distance here
    * @tparam B
    * @return
    */
  def map[B](f: A => B): StateM[S, B] = {
    println(">>> map")
    flatMap(a => StateM.point(f(a))) //Creates a new StateM with run s => s, f(a)
  }
}

object StateM {
  //type is named as B to be inline with map() API
  def point[S, B](v: B): StateM[S, B] = StateM(run = s => (s, v))
}


case class GolfState(distance: Int)

object StateMonad {

  def main(args: Array[String]): Unit = {


    val beginningStateM = GolfState(0)

    /**
      * swing() is a function that takes the state info as a int and creates a State Monad,
      * whicn then can be run to compute new state
      * @param distance
      * @return
      */
    def swing(distance: Int): StateM[GolfState, Int] = StateM { (s: GolfState) => //run: S => (S, A)
      println(">>> run/swing triggered: distance: " + distance + " state: " + s)
      val newDistance = s.distance + distance
      println("<<< run/swing: " + (GolfState(newDistance), newDistance) + "")
      (GolfState(newDistance), newDistance)
    }

    val firstHit = swing(distance=10) //Lets say we moved the ball by 10 meters
//    println("firstHit: " + firstHit)
    // State doesn't change unless we ask it to do so
    val afterFirstHit = firstHit.run(beginningStateM)
    println("afterFirstHit " + afterFirstHit)

    println("\n-------------------------------------------------\n")

    println("about to make second hit... ")
    //Lets move the ball by another 10 meters
    val secondHit = firstHit.flatMap{_ =>
      println("secondhit by calling swing inside the flatMap")
      swing(20)} //flatMap[B](g: A => StateM[S, B]): StateM[S, B]
    println("Lets move the ball by another 20 meters " + secondHit.run(beginningStateM))

    println("\n-------------------------------------------------\n")

    //Lets move the ball by another 10 meters
    val thirdHit = secondHit.map(distance => distance + 30)
    println("Lets move the ball by another 30 meters " + thirdHit.run(beginningStateM))

    println("\n-------------------------------------------------\n")

    val stateWithNewDistance: StateM[GolfState, Int] = for {
      _             <- swing(distance=10)
      _             <- swing(distance=20)
      totalDistance <- swing(distance=30)
    } yield totalDistance

    // THE ACTION BEGINS

    // `run` is like `unsafeRunSync` in the Cats `IO` monad
    val result: (GolfState, Int) = stateWithNewDistance.run(beginningStateM)

    println(s"GolfState:      ${result._1}")  //GolfState(60)
    println(s"Total Distance: ${result._2}") //60

    println("\n-------------------------------------------------\n")

    //compiler generated : scalac -Xprint:parse src/main/scala/io/dhiraa/StateMonad.scala
    //flatMap[B](g: A => StateM[S, B]): StateM[S, B]
    val stateWithNewDistance1: StateM[GolfState, Int] = swing(distance=10).flatMap { distance : Int =>
      println("g on flatmap annonymous function with distance " + distance)
      swing(distance = 20).flatMap {distance  =>
        println("g on flatmap with distance " + distance)
        swing(distance = 30).map{totalDistance =>
          println("g on flatmap with totalDistance " + totalDistance)
          totalDistance}
      }
    }

    val result1 = stateWithNewDistance1.run(beginningStateM)

    println(s"GolfState:      ${result1._1}")  //GolfState(60)
    println(s"Total Distance: ${result1._2}") //60

  }
}

