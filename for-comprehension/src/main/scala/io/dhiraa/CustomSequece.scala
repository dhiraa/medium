package io.dhiraa

import scala.collection.mutable.ArrayBuffer

abstract class CustomClass[A] {
  def foreach(b: A => Unit): Unit
  def map[B](f: A => B): CustomClass[B]
  def flatMap[B](f: A => CustomClass[B]): CustomClass[B]
  def withFilter(p: A => Boolean): CustomClass[A]
}

/**
  *
  * @param initialElems
  * @tparam A
  */
case class Sequence[A](private val initialElems: A*) {
  //To make things simple replace all A & B's with Int

  // this is a test, don't do this at production
  //following two lines goes into class constructor
  private val elems: ArrayBuffer[A] = ArrayBuffer[A]()
  // initialize
  elems ++= initialElems

  def foreach(block: A => Unit): Unit = {
    elems.foreach(block)
  }

  def map[B](f: A => B): Sequence[B] = {
    val abMap = elems.map(f)
    Sequence(abMap: _*) //_* -> varargs; _ -> says infer the type, * says it is a sequence
  }

  /**
    * val a = List( List(1,2), List(3,4) )
    * a.flatten //res0: List[Int] = List(1, 2, 3, 4)
    * @param seqOfSeq
    * @tparam B
    * @return
    */
  private def flattenLike[B](seqOfSeq: Sequence[Sequence[B]]): Sequence[B] = {
    var xs = ArrayBuffer[B]()
    for (listB: Sequence[B] <- seqOfSeq) {
      for (e <- listB) {
        xs += e
      }
    }
    Sequence(xs: _*)
  }

  def flatMap[B](f: A => Sequence[B]): Sequence[B] = {
    val mapRes: Sequence[Sequence[B]] = map(f) //map
//    mapRes.foreach(println)
    flattenLike(mapRes) //flatten
  }

  def withFilter(p: A => Boolean): Sequence[A] = {
    val tmpArrayBuffer = elems.filter(p)
    Sequence(tmpArrayBuffer: _*)
  }

}

case class Person(name: String)

//scalac -Xprint:parse src/main/scala/io/dhiraa/CustomSequece.scala

object CustomSequece {

  def main(args: Array[String]): Unit = {

    val ints = Sequence(1,2,3)

    val test = for {
      i <- ints
    } yield i*2
    //val test = ints.map(((i) => i.$times(2)));
    //As you can see yield i*2 goes as lambda function

    test.foreach(println)

    println("-----------------------------------------------------")


    val myFriends = Sequence(
      Person("Adam"),
      Person("David"),
      Person("Frank")
    )

    val adamsFriends = Sequence(
      Person("Nick"),
      Person("David"),
      Person("Frank")
    )

    val mutualFriends = for {
      myFriend <- myFriends //  flatmap
      adamsFriend <- adamsFriends
      if myFriend.name == adamsFriend.name //withFilter anonymous function
    } yield myFriend //map on filtered adamsFriend list
    println("--------------------------")

    mutualFriends.foreach(println)
    //Person(David)
    //Person(Frank)

    println("-----------------------------------------------------")

    //compiler generated
    val mutualFriends1 = myFriends.flatMap( myFriend =>
      adamsFriends.withFilter( adamsFriend =>
        myFriend.name.$eq$eq(adamsFriend.name)
      ).map( adamsFriend => myFriend)
    )
    println("--------------------------")
    mutualFriends1.foreach(println)
  }


}
